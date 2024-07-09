import json

import torch
import nni.retiarii.nn.pytorch as nn

import sys


def get_parameters(model, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag:
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag:
                yield param
    else:
        raise ValueError('do not support: %s' % mode)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def build_activation(act_func, inplace=True, num_parameters=1):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'prelu':
        return nn.PReLU(num_parameters=num_parameters)
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim) + 1e-8
        pred = torch.log(pred)
        num_classes = pred.size(self.dim)
        # pred = torch.where(torch.eq(pred, torch.zeros_like(pred)), torch.full_like(pred, 1.), pred)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_nas_network(args, class_flag=False):
    if args.net == 'searchsupermodel50':
        from models.supermodel import supermodel50
        net = supermodel50
    elif args.net == 'searchsupermodel8':
        from models.supermodel import supermodel8
        net = supermodel8
    elif args.net == 'searchsupermodel16':
        from models.supermodel import supermodel16
        net = supermodel16
    elif args.net == 'searchcifarsupermodel16':
        from models.supermodel import cifarsupermodel16
        net = cifarsupermodel16
    elif args.net == 'searchcifarsupermodel22':
        from models.supermodel import cifarsupermodel22
        net = cifarsupermodel22
    elif args.net == 'searchcifarsupermodel26':
        from models.supermodel import cifarsupermodel26
        net = cifarsupermodel26
    elif args.net == 'searchcifarsupermodel50':
        from models.supermodel import cifarsupermodel50
        net = cifarsupermodel50
    elif args.net == 'searchcifarsupermodel101':
        from models.supermodel import cifarsupermodel101
        net = cifarsupermodel101
    elif args.net == 'searchcifarsupermodel80':
        from models.supermodel import cifarsupermodel80
        net = cifarsupermodel80
    elif args.net == 'searchcifarsupermodel152':
        from models.supermodel import cifarsupermodel152
        net = cifarsupermodel152
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if not class_flag:
        if args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'cifar10':
            n_classes = 10
        elif args.dataset == 'imagenet':
            n_classes = 1000
        else:
            n_classes = 1000
        net = net(pretrained=args.pretrained, num_classes=n_classes)

    return net


def generate_arch(checkpoint_prob_path, output_path=None):
    if output_path is None:
        output_path = checkpoint_prob_path.rstrip('.prob') + '.tmp'
    with open(checkpoint_prob_path, 'r') as f, open(output_path, 'w') as out:
        prob = json.load(f)
        arch = dict()
        for key in prob.keys():
            arch[key] = prob[key].index(max(prob[key]))
        json.dump(arch, out)
        return arch


class BinaryPReLu(nn.Module):
    def __init__(self):
        super(BinaryPReLu, self).__init__()
        self.relu = nn.PReLU()

    def forward(self, x):
        weight = self.relu.weight
        # torch.clamp(weight, 0, 1)
        ones = torch.ones_like(weight)
        zeros = torch.zeros_like(weight)
        self.relu.weight = torch.where(weight <= 0.5, zeros, ones)
        return self.relu(x)


def reproduce_model(model, threshold=0.5):
    for name, param in model.named_parameters():
        if 'alpha' in name:
            values, _ = torch.max(param.data, dim=0)
            threshold = values * threshold
            alpha = torch.where(param.data > threshold, torch.ones_like(param.data), torch.zeros_like(param.data))
            param.data.copy_(alpha)
            param.requires_grad = False
        elif 'beta' in name:
            values, idx = torch.max(param.data, dim=0)
            param.data[idx] = 1
            param.data[1-idx] = 0
            param.requires_grad = False
