# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import os
import logging
import pickle
import time
from models.supermodel import _SampleLayer
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.oneshot.interface import BaseOneShotTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device

from nas.estimator import Estimator, _get_module_with_type
from utils.tools import get_relu_count

_logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


class ArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = x.detach()
        detached_x.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None


class ProxylessLayerChoice(nn.Module):
    def __init__(self, ops):
        super(ProxylessLayerChoice, self).__init__()
        self.ops = nn.ModuleList(ops)
        self.alpha = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self._binary_gates = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self.sampled = None

    def forward(self, *args):
        def run_function(ops, active_id):
            def forward(_x):
                return ops[active_id](_x)

            return forward

        def backward_function(ops, active_id, binary_gates):
            def backward(_x, _output, grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for k in range(len(ops)):
                        if k != active_id:
                            out_k = ops[k](_x.data)
                        else:
                            out_k = _output.data
                        grad_k = torch.sum(out_k * grad_output)
                        binary_grads[k] = grad_k
                return binary_grads

            return backward

        assert len(args) == 1
        x = args[0]
        return ArchGradientFunction.apply(
            x, self._binary_gates, run_function(self.ops, self.sampled),
            backward_function(self.ops, self.sampled, self._binary_gates)
        )

    def resample(self):
        probs = F.softmax(self.alpha, dim=-1)
        sample = torch.multinomial(probs, 1)[0].item()
        self.sampled = sample
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self.alpha.grad is None:
                self.alpha.grad = torch.zeros_like(self.alpha.data)
            probs = F.softmax(self.alpha, dim=-1)
            for i in range(len(self.ops)):
                for j in range(len(self.ops)):
                    self.alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])

    def export(self):
        return torch.argmax(self.alpha).item()

    def export_prob(self):
        return F.softmax(self.alpha, dim=-1)


class ProxylessInputChoice(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Input choice is not supported for ProxylessNAS.')


class ProxylessTrainer(BaseOneShotTrainer):
    """
    Proxyless trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    warmup_epochs : int
        Number of epochs to warmup model parameters.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    grad_reg_loss_type: string
        Regularization type to add hardware related loss, allowed types include
        - ``"mul#log"``: ``regularized_loss = (torch.log(expected_latency) / math.log(self.ref_latency)) ** beta``
        - ``"add#linear"``: ``regularized_loss = reg_lambda * (expected_latency - self.ref_latency) / self.ref_latency``
        - None: do not apply loss regularization.
    grad_reg_loss_params: dict
        Regularization params, allowed params include
        - ``"alpha"`` and ``"beta"`` is required when ``grad_reg_loss_type == "mul#log"``
        - ``"lambda"`` is required when ``grad_reg_loss_type == "add#linear"``
    applied_hardware: string
        Applied hardware for to constraint the model's latency. Latency is predicted by Microsoft
        nn-Meter (https://github.com/microsoft/nn-Meter).
    dummy_input: tuple
        The dummy input shape when applied to the target hardware.
    ref_latency: float
        Reference latency value in the applied hardware (ms).
    """

    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, warmup_epochs=0,
                 batch_size=64, workers=4, device=None, log_frequency=None,
                 arc_learning_rate=1.0E-3,
                 grad_reg_loss_type=None, grad_reg_loss_params=None,
                 applied_hardware=None, dummy_input=(1, 3, 224, 224),
                 checkpoint_path=None, 
                 expect_latency_rate=0.2,
                 offline_upper=0.4,
                 teacher=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency

        self.checkpoint_path = checkpoint_path

        # Hyper parameters
        self.reg_lambda = 0.005
        self.reg_l1_var = 0.001

        # knowledge distillation
        self.teacher = teacher
        if teacher is not None:
            self.teacher = torch.nn.DataParallel(teacher)
            self.teacher.to(self.device)
            self.teacher.eval()
        self.temp = 4
        self.alpha = 0.3
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')

        if not applied_hardware:
            applied_hardware = {"nonlinear": 0.228753397836538, "linear": 3.55361855670103E-06}
        self.latency_estimator = Estimator(applied_hardware, model, offline_upper, dummy_input)

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params
        
        # get expected latency
        linear, nonlinear = self.latency_estimator.cal_expected_latency(self.model, max_latency=True)
        self.ref_latency = (linear + nonlinear) * expect_latency_rate
        
        # initialize model
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.nas_modules = []
        replace_layer_choice(self.model, ProxylessLayerChoice, self.nas_modules)
        replace_input_choice(self.model, ProxylessInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        self.obj_path = self.checkpoint_path.rstrip('.json') + '.o'
        if os.path.exists(self.obj_path):
            self._load_from_checkpoint(self.obj_path)
        self.optimizer = optimizer
        # we do not support deduplicate control parameters with same label (like DARTS) yet.
        self.ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], arc_learning_rate,
                                           weight_decay=0.01, betas=(0, 0.999), eps=1e-8)
        self._init_dataloader()
        self._init_logger()

    def _init_dataloader(self):
        n_train = len(self.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers)

    def _init_logger(self):
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.path.dirname('logs/')
        log_path = os.path.join(log_path, self.reg_loss_type)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = os.path.join(log_path, rq + '.log')
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
            val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

            if epoch >= self.warmup_epochs:
                # 1) train architecture parameters
                for _, module in self.nas_modules:
                    module.resample()
                self.ctrl_optim.zero_grad()
                logits, loss = self._logits_and_loss_for_arch_update(val_X, val_y)
                loss.backward()
                for _, module in self.nas_modules:
                    module.finalize_grad()
                self.ctrl_optim.step()

            # 2) train model parameters
            for _, module in self.nas_modules:
                module.resample()
            self.optimizer.zero_grad()
            logits, loss = self._logits_and_loss_for_weight_update(trn_X, trn_y)
            loss.backward()
            self.optimizer.step()
            metrics = self.metrics(logits, trn_y)
            metrics["loss"] = loss.item()
            if self.latency_estimator:
                metrics["nonlinear_count"] = self._export_latency()
            meters.update(metrics)
            print("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)

    def _get_arch_relu(self):
        current_architecture_prob = []
        for module_name, module in self.nas_modules:
            probs = module.export_prob().detach()
            current_architecture_prob.append(probs)
        return current_architecture_prob

    def _logits_and_loss_for_arch_update(self, X, y):
        """ return logits and loss for architecture parameter update """
        logits = self.model(X)
        ce_loss = self.loss(logits, y)
        if not self.latency_estimator:
            return logits, ce_loss

        linear, nonlinear = self.latency_estimator.cal_expected_latency(self.model)
        expected_latency = linear + nonlinear
        
        # loss_balance = abs(linear/nonlinear) + abs(nonlinear/linear)
        loss_balance = abs(linear - nonlinear)/nonlinear

        if self.reg_loss_type == 'add#linear':
            reg_loss = self.reg_lambda * abs((expected_latency - self.ref_latency) / self.ref_latency)
            return logits, ce_loss + reg_loss + loss_balance * self.reg_l1_var
        elif self.reg_loss_type is None:
            return logits, ce_loss
        else:
            raise ValueError(f'Do not support: {self.reg_loss_type}')

    def _logits_and_loss_for_weight_update(self, X, y):
        ''' return logits and loss for weight parameter update '''
        logits = self.model(X)
        loss = self.loss(logits, y)
        # teacher model
        if self.teacher is not None:
            with torch.no_grad():
                teachers_preds = self.teacher(X)
            # calculate soft_loss
            distillation_loss = self.soft_loss(
                F.log_softmax(logits / self.temp, dim=1),
                F.softmax(teachers_preds / self.temp, dim=1)
            ) * self.temp * self.temp
            # add hard_loss and soft_loss
            loss = loss + distillation_loss 
        return logits, loss

    def _export_latency(self):
        linear, nonlinear = self.latency_estimator.cal_expected_latency(self.model)
        expected_latency = linear + nonlinear
        return expected_latency

    def _load_from_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            state_dict = pickle.load(f)
            self.model.load_state_dict(state_dict)
            _logger.info("load checkpoint from %s", checkpoint_path)

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)
            print('finish epoch ', i)
            if self.checkpoint_path is not None:
                json.dump(self.export_prob(), open(self.checkpoint_path + '.prob', 'w'))
                with open(self.obj_path, 'wb') as f:
                    pickle.dump(self.model.state_dict(), f)

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result

    @torch.no_grad()
    def export_prob(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export_prob().tolist()
        return result
