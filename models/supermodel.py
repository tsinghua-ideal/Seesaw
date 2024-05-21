from collections import OrderedDict
from functools import reduce
import torch
from torch import Tensor
import torch.nn.functional as F
import nni.nas.nn.pytorch as nn
from typing import Tuple, Type, Any, Callable, Union, List, Optional
from models.ops import OPS, DropPath_, ZeroLayer
import copy
from utils.tools import calculate_flops


class _SampleLayer(nn.Module):
    # all ops
    SAMPLE_OPS = ['none', 
                  'avg_pool_3x3', 'avg_pool_5x5', 'avg_pool_7x7', 
                  'skip_connect', 
                  'conv_1x1', 'conv_3x3', 'conv_5x5', 'conv_7x7', 
                  'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 
                  'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7', 
                  'dil_sep_conv_3x3', 'dil_sep_conv_5x5', 'dil_sep_conv_7x7', 
                  'group_8_conv_3x3', 'group_8_conv_5x5', 'group_8_conv_7x7', 
                  'conv_3x1_1x3', 'conv_5x1_1x5', 'conv_7x1_1x7', 
                  'van_conv_3x3', 'van_conv_5x5', 'van_conv_7x7']

    def __init__(
            self,
            inplanes: int,
            label: str,
            clamp=False
    ) -> None:
        super(_SampleLayer, self).__init__()
        self.clamp = clamp
        # extract feature from low dimension
        self.paths = nn.ModuleList([nn.Sequential(OPS[op](inplanes, 1, True), DropPath_()) for op in self.SAMPLE_OPS])
        self.alpha = nn.Parameter(torch.rand(len(self.SAMPLE_OPS)) * 1E-3)
        self.nonlinear = nn.LayerChoice([nn.Identity(), nn.ReLU()])
        self.softmax = nn.Softmax(dim=-1)
        self.flops_list = self._stat_flops()
        self.output_shape = None

    def _stat_flops(self):
        flops = []
        for module in self.paths:
            flops.append(calculate_flops(module))
        return flops

    def replace_zero_layers(self, threshold=1e-2):
        betas = self.softmax(self.alpha)
        replace_indices = [index for index, weight in enumerate(betas) if weight < threshold]
        for index in replace_indices:
            self.paths[index] = ZeroLayer()
            betas[index] = 1
            
        non_zero_count = len(self.paths)
        for idx, module in enumerate(self.paths):
            if isinstance(module, ZeroLayer):
                betas[idx] = 1
                non_zero_count -= 1
        if non_zero_count > 3:
            min_index = torch.argmin(betas)
            self.paths[min_index] = ZeroLayer()
                         
    def forward(self, x: Tensor) -> Tensor:
        weights = self.softmax(self.alpha)
        out = None
        for idx, _ in enumerate(self.paths):
            if out is None:
                out = self.paths[idx](x) * weights[idx]
            else:
                out = out + self.paths[idx](x) * weights[idx]
        out = self.nonlinear(out)    
        if self.output_shape is None:
            self.output_shape = out.size()
        return out


class SampleBlock(nn.ModuleDict):
    def __init__(
            self,
            num_layers: int,
            inplanes: int,
    ) -> None:
        super(SampleBlock, self).__init__()
        layer = nn.Sequential(
            nn.Conv2d(inplanes, inplanes * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(inplanes * 2),
        )
        self.add_module('upsamplelayer', layer)
        for i in range(num_layers):
            # FIXME: nn.InputChoice maybe needed
            layer = _SampleLayer(inplanes * 2, 'sampleunit')
            self.add_module('samplelayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = init_features
        for _, layer in self.items():
            features = layer(features)
        return features


class AggregateBlock(nn.Module):
    def __init__(
            self,
            inplanes: list,
            outplanes: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(AggregateBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layers = nn.ModuleList()
        compensation = 1
        config = inplanes[:]
        config.reverse()
        for idx, inp in enumerate(config):
            self.layers.insert(0, nn.Sequential(
                nn.Conv2d(inp, outplanes, kernel_size=1, stride=compensation, bias=False),
                norm_layer(outplanes)
            ))
            compensation = 2 ** (idx)
        self.nonlinear = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # self.nonlinear = nn.LayerChoice([nn.AvgPool2d(kernel_size=2, stride=2), 
        #                                  nn.Sequential(
        #                                     nn.MaxPool2d(kernel_size=2, stride=2),
        #                                     nn.ReLU())
        # ])
        self.alpha = nn.Parameter(torch.rand(len(self.layers)) * 1E-3)
        self.softmax = nn.Softmax(dim=-1)
        self.flops_list = self._stat_flops()
        self.output_shape = None
        
    def _stat_flops(self):
        flops = []
        for module in self.layers:
            flops.append(calculate_flops(module))
        return flops

    def freeze(self, topk=1):
        if topk == -1:
            topk = int(len(self.layers)/2)
        elif topk > len(self.layers):
            topk = len(self.layers) 
        _, indices = torch.topk(self.alpha, topk)
        y = torch.zeros_like(self.alpha)
        y[indices] = 1
        print(y)
        self.alpha = nn.Parameter(y)
        self.alpha.requires_grad_(False)

    def forward(self, x: List) -> Tensor:
        weights = self.softmax(self.alpha)
        out = None
        for idx, _ in enumerate(self.layers):
            if out is None:
                out = self.layers[idx](x[idx]) * weights[idx]
            else:
                out = out + self.layers[idx](x[idx]) * weights[idx]

        out = self.nonlinear(out)
        if self.output_shape is None:
            self.output_shape = out.size()
        return out


# @model_wrapper
class Supermodel(nn.Module):
    def __init__(
            self,
            dataset: str = 'imagenet',
            num_init_features: int = 32,
            block_config: Tuple[int, int, int, int] = (2, 2, 2, 2),
            num_classes: int = 1000
    ) -> None:

        super(Supermodel, self).__init__()

        # First convolution
        if dataset == 'imagenet':
            init_stride = 4
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=init_stride,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
            ]))

        self.samples = nn.ModuleList()
        self.aggeregate = nn.ModuleList()
        channels = [num_init_features]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # FIXME: Maybe unreasonable without channel adding
            block = SampleBlock(
                num_layers=num_layers,
                inplanes=num_features
            )
            self.samples.append(block)
            num_features = num_features * 2
            if i != len(block_config) - 1:
                channels.append(num_features)
                trans = AggregateBlock(channels, num_features)
                self.aggeregate.append(trans)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_total_count(self, max_latency=False):
        total_flops = 0
        relu_count = 0
        if max_latency:
            for m in self.modules():
                if isinstance(m, _SampleLayer):
                    relu_count += abs(reduce(lambda x, y: x * y, m.output_shape))
                    total_flops += sum(m.flops_list)
                elif isinstance(m, AggregateBlock):
                    relu_count += abs(reduce(lambda x, y: x * y, m.output_shape))
                    total_flops += sum(m.flops_list)
        else:
            for m in self.modules():
                if isinstance(m, _SampleLayer):
                    if isinstance(m.nonlinear, nn.LayerChoice):
                        probs = m.nonlinear.export_prob()
                    elif isinstance(m.nonlinear, nn.Identity):
                        probs = [1, 0]
                    else:
                        probs = [0, 1]
                    relu_count += abs(reduce(lambda x, y: x * y, m.output_shape)) * probs[1]
                    w = m.softmax(m.alpha)
                    for i, flops in enumerate(m.flops_list):
                        total_flops += w[i] * flops
                elif isinstance(m, AggregateBlock):
                    if isinstance(m.nonlinear, nn.LayerChoice):
                        probs = m.nonlinear.export_prob()
                    elif isinstance(m.nonlinear, nn.Identity):
                        probs = [1, 0]
                    else:
                        probs = [0, 1]
                    relu_count += abs(reduce(lambda x, y: x * y, m.output_shape)) * probs[1]
                    w = m.softmax(m.alpha)
                    for i, flops in enumerate(m.flops_list):
                        total_flops += w[i] * flops
        return total_flops, relu_count
    
    def get_total_flops(self, max_latency=False):
        total_flops = 0
        if max_latency:
            for m in self.modules():
                if isinstance(m, _SampleLayer):
                    total_flops += sum(m.flops_list)
                elif isinstance(m, AggregateBlock):
                    total_flops += sum(m.flops_list)
        else:
            for m in self.modules():
                if isinstance(m, _SampleLayer):
                    w = m.softmax(m.alpha).tolist()
                    for i, prob in enumerate(w):
                        total_flops += prob * m.flops_list[i]
                elif isinstance(m, AggregateBlock):
                    w = m.softmax(m.alpha).tolist()
                    for i, prob in enumerate(w):
                        total_flops += prob * m.flops_list[i]
        return total_flops
    
    # def get_flops_count_list(self):
    #     flops_list = []              
    #     counts_list = []
    #     for m in self.modules():
    #         if isinstance(m, _SampleLayer) or isinstance(m, AggregateBlock):
    #             probs = m.nonlinear.export_prob().detach()
    #             counts_list.append((probs[1], abs(reduce(lambda x, y: x * y, m.output_shape))))
    #             w = m.softmax(m.alpha).tolist()
    #             flops_list.append((w, m.flops_list))
    #     return flops_list, counts_list

    def forward(self, x: Tensor) -> Tensor:
        features = [self.features(x)]
        for idx in range(len(self.aggeregate)):
            features.append(self.samples[idx](features[-1]))
            features[-1] = self.aggeregate[idx](features)
        out = self.samples[-1](features[-1])

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def supermodel16(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(2, 4), num_classes=num_classes)


def supermodel8(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(2, 2, 2), num_classes=num_classes)


def supermodel50(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(3, 4, 6, 3), num_classes=num_classes)


def cifarsupermodel16(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2, 2, 2), num_classes=num_classes)


def supermodel101(num_classes: int = 1000, pretrained: bool = False):
    return Supermodel(block_config=(3, 4, 23, 3), num_classes=num_classes)


def cifarsupermodel22(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2, 2), num_classes=num_classes)


def cifarsupermodel26(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(2, 2), num_classes=num_classes)


def cifarsupermodel50(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(3, 4, 6, 3), num_classes=num_classes)


def cifarsupermodel101(num_classes: int = 100, pretrained: bool = False):
    return Supermodel(dataset='cifar', block_config=(3, 4, 23, 3), num_classes=num_classes)