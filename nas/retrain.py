import os
import time
import math
from datetime import timedelta
import torch
from torch import nn as nn
import torch.nn.functional as F
from nni.retiarii.oneshot.pytorch.utils import AverageMeter
from fvcore.nn import FlopCountAnalysis

from torch.utils.tensorboard import SummaryWriter
from utils import config
from utils.tools import get_relu_count
from models.supermodel import _SampleLayer
from nas.estimator import Estimator

import logging
logging.basicConfig(level=logging.ERROR)


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Retrain:
    def __init__(self, model, optimizer, device, data_provider, n_epochs, export_path, teacher=None, spatial_trainable=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = data_provider.train
        self.valid_loader = data_provider.valid
        self.test_loader = data_provider.test
        self.data_shape = data_provider.data_shape
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        # change it while training
        # print relu_count
        print('The input data shape is: ', self.data_shape)
        self.spatial_trainable = spatial_trainable 
        print('The spatial_training flag is: ', self.spatial_trainable)     
        count = get_relu_count(self.model, self.data_shape)
        print('The relu count of current model is: ', count)
        self.model.to(self.device)
        
        applied_hardware = {"offline_nonlinear": 0.053842398, "online_nonlinear": 0.174911, "linear": 3.55361855670103E-06}
        self.applied_hardware = applied_hardware
        self.latency_estimator = Estimator(applied_hardware, self.model, (1,)+self.data_shape)
        
        fixed_pruning = False
        
        if spatial_trainable:
            print('Spatial training')
            self.export_path = export_path.rstrip('.pth') + '.o'   
            nonlinear = get_relu_count(self.model, self.data_shape) * applied_hardware['online_nonlinear']
            self.nonlinear_cost = nonlinear
            # self.applied_hardware = {"nonlinear": 0.174911, "offline_linear": 3.44E-05, "online_linear": 7.01E-08}
            # self.nonlinear_cost = get_relu_count(self.model, self.data_shape) * self.applied_hardware['nonlinear']
        elif fixed_pruning:
            self.export_path = export_path.rstrip('.pth') + '-' + str(count) + '.pth-fixed'
            print('The loading path is: ', self.export_path)
            if os.path.exists(self.export_path):
                st = torch.load(self.export_path)
                self.model.load_state_dict(st, strict=False)    
            print('Pruning linear operators ...')
            for _, module in model.named_modules():
                if isinstance(module, _SampleLayer):
                    module.remove_layer()
            online = get_relu_count(self.model, self.data_shape) * applied_hardware['online_nonlinear']
            self.model.eval()
            if torch.cuda.is_available():
                flops_counter = FlopCountAnalysis(self.model, torch.rand((1,)+self.data_shape).cuda())
            else:
                flops_counter = FlopCountAnalysis(self.model, torch.rand((1,)+self.data_shape).cpu())
            self.model.train()
            offline = flops_counter.total() * applied_hardware['linear'] + count * applied_hardware['offline_nonlinear']
            print(f'offline cost: {offline}, online cost: {online}')
        else: 
            self.export_path = export_path.rstrip('.pth') + '-' + str(count) + '.pth-var'
            print('The loading path is: ', self.export_path)
            if os.path.exists(self.export_path):
                st = torch.load(self.export_path)
                self.model.load_state_dict(st, strict=False)    
            print('Pruning linear operators ...')
            old_linear = 0
            while True:
                for _, module in self.model.named_modules():
                    if isinstance(module, _SampleLayer):
                        module.replace_zero_layers(1e-4)
                nonlinear = get_relu_count(self.model, self.data_shape) * applied_hardware['online_nonlinear']
                self.model.eval()
                if torch.cuda.is_available():
                    flops_counter = FlopCountAnalysis(self.model, torch.rand((1,)+self.data_shape).cuda())
                else:
                    flops_counter = FlopCountAnalysis(self.model, torch.rand((1,)+self.data_shape).cpu())
                self.model.train()
                linear = flops_counter.total() * applied_hardware['linear'] + count * applied_hardware['offline_nonlinear']
                print(f'offline cost: {linear}, online cost: {nonlinear}')
                if linear < nonlinear or abs(linear - nonlinear)/nonlinear < 1e-6 or old_linear == linear:
                    break
                old_linear = linear
            print('Finish pruning linear operators ...')
        print('The export path is: ', self.export_path)
        self.model.to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            flops_counter = FlopCountAnalysis(self.model, torch.rand((1,)+self.data_shape).cuda())
        else:
            flops_counter = FlopCountAnalysis(self.model, torch.rand((1,)+self.data_shape).cpu())
        self.model.train()
        print('The flops of current model is: ', flops_counter.total())
        
        # knowledge distillation
        self.teacher = teacher
        self.temp = 4
        self.alpha = 0.3
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')

        self.writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, self.export_path, config.TIME_NOW))

    def run(self):
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        if self.teacher is not None:
            self.teacher = torch.nn.DataParallel(self.teacher)
            self.teacher.to(self.device)
            self.teacher.eval()
        # train
        self.train()
        # validate
        self.validate(is_test=False)
        # test
        self.validate(is_test=True)
        
        self.writer.close()

    def train_one_epoch(self, adjust_lr_func, train_log_func, label_smoothing=0.1):
        batch_time = AverageMeter('batch_time')
        data_time = AverageMeter('data_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        self.model.train()
        end = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            if label_smoothing > 0:
                students_loss = cross_entropy_with_label_smoothing(output, labels, label_smoothing)
            else:
                students_loss = self.criterion(output, labels)
            # teacher model
            if self.teacher is not None:
                with torch.no_grad():
                    teachers_preds = self.teacher(images)
                # calculate soft_loss
                distillation_loss = self.soft_loss(
                    F.log_softmax(output / self.temp, dim=1),
                    F.softmax(teachers_preds / self.temp, dim=1)
                ) * self.temp * self.temp
                # add hard_loss and soft_loss
                loss = students_loss + distillation_loss 
            else:
                loss = students_loss
            if self.spatial_trainable:
                l1_reg = 0
                for _, module in self.model.named_modules():
                    if isinstance(module, _SampleLayer):
                        l1_reg += torch.var(module.alpha)
                loss -= l1_reg * 1e-4
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.model.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 or i + 1 == len(self.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                # print(batch_log)
        return top1, top5

    def train(self, validation_frequency=1):
        best_acc = 0
        nBatch = len(self.train_loader)

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
                self.writer.add_scalar('Train/loss', losses.avg, epoch_ + 1)

                batch_log = 'Train [{0}][{1}/{2}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                            'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                            'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                    format(epoch_ + 1, i, nBatch - 1,
                        batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
                batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                batch_log += '\tlr {lr:.5f}'.format(lr=lr)
                return batch_log
        
        def adjust_learning_rate(n_epochs, optimizer, epoch, batch=0, nBatch=None):
            """ adjust learning of a given optimizer and return the new learning rate """
            # cosine
            T_total = n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            new_lr = 0.5 * 0.05 * (1 + math.cos(math.pi * T_cur / T_total))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr

        for epoch in range(self.n_epochs):
            # print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: adjust_learning_rate(self.n_epochs, self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            # seconds_left = int((self.n_epochs - epoch - 1) * time_per_epoch)
            # print('Time per epoch: %s, Est. complete in: %s' % (
            #     str(timedelta(seconds=time_per_epoch)),
            #     str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False)
                is_best = val_acc > best_acc
                best_acc = max(best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.n_epochs, val_loss, val_acc, best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
                    format(val_acc5, top1=train_top1, top5=train_top5)
                # print(val_log)
                self.writer.add_scalar('Test/Average loss', val_loss, epoch + 1)
                self.writer.add_scalar('Test/Accuracy', val_acc, epoch + 1)
            else:
                is_best = False
            if is_best:
                print('Best accuracy {}'.format(best_acc))
                torch.save(self.model.module.state_dict(), self.export_path)

    def validate(self, is_test=True):
        if is_test:
            data_loader = self.test_loader
        else:
            data_loader = self.valid_loader
        self.model.eval()
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = self.model(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    # print(test_log)
        return losses.avg, top1.avg, top5.avg
