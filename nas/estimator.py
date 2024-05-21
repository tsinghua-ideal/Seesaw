import copy
from functools import reduce
import logging
import sys
from collections import OrderedDict

import torch
import nni.nas.nn.pytorch as nn
from utils.tools import model_latency, size2memory

_logger = logging.getLogger(__name__)


def _get_module_with_type(root_module, type_name, modules):
    if modules is None:
        modules = []
    if not isinstance(type_name, (list, tuple)):
        type_name = [type_name]

    def apply(m):
        for name, child in m.named_children():
            for t_name in type_name:
                if isinstance(child, t_name):
                    modules.append(child)
                else:
                    apply(child)

    apply(root_module)
    return modules


class NonlinearLatencyEstimator:
    def __init__(self, hardware, model, dummy_input=(1, 3, 224, 224)):
        _logger.info(f'Get latency predictor for applied hardware: {hardware}.')
        self.hardware = hardware
        self.in_size = dummy_input
        self.block_latency_table, self.total_latency = model_latency(model, dummy_input[1:], hardware)
        self.non_ops = _get_module_with_type(model, nn.Hardswish, [])
        self.choices = _get_module_with_type(model, nn.LayerChoice, [])
        self._refresh_table(_get_module_with_type(model, nn.LayerChoice, []))
        print('NonlinearLatencyEstimator initialized...')

    def _refresh_table(self, choices):
        idx = 0
        new_table = copy.copy(self.block_latency_table)
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                in_size = self.block_latency_table[name]['input_shape']
                layer_table = OrderedDict()
                if len(choices) <= idx:
                    continue
                for i, choice in enumerate(list(choices[idx])):
                    table, lat = model_latency(choice, in_size[:], self.hardware)
                    layer_table[str(i)] = table
                    if i == 0:
                        self.total_latency -= lat
                        num_choices = len(table)
                        for j, key in enumerate(table):
                            rm_key = '_'.join(key.split('_')[:-1] + [str(int(name.split('_')[-1]) + j - num_choices)])
                            new_table.pop(rm_key)
                        continue
                new_table[name] = layer_table
                idx += 1
        self.block_latency_table = copy.copy(new_table)

    def _cal_nonlinear_count(self, cur_arch_prob):
        total = 0.0
        for name in self.block_latency_table.keys():
            if name.startswith('LayerChoice'):
                layer_choice_prob = cur_arch_prob.pop(0)
                table = self.block_latency_table[name]
                for idx, key in enumerate(table):
                    to = 0.0
                    for _, op in enumerate(table[key]):
                        if op.startswith('ReLU') or op.startswith('Hardswish'):
                            to += abs(reduce(lambda x, y: x * y, table[key][op]["output_shape"]))
                    total += to * layer_choice_prob[idx]
            elif name.startswith('ReLU') or name.startswith('Hardswish'):
                total += abs(reduce(lambda x, y: x * y, table[key][op]["output_shape"]))
        return total

    def cal_expected_latency(self, cur_arch_prob):
        lat = self._cal_nonlinear_count(cur_arch_prob)
        return lat


class Estimator:
    def __init__(self, hardware, model, offline_upper=0.5, dummy_input=(1, 3, 224, 224)):
        _logger.info(f'Get latency predictor for applied hardware: {hardware}.')
        self.hardware = hardware
        model_copy = copy.copy(model)
        model_copy.eval()
        model_copy = model_copy.to('cpu')
        dummy_input = torch.randn(dummy_input).to('cpu')
        model_copy(dummy_input)
        self.upper = offline_upper
        # self.offline_upper = self._get_init_latency(model=model, upper=offline_upper)
        print('Estimator initialized...')
        
    def cal_expected_latency(self, model, max_latency=False):
        if isinstance(model, torch.nn.DataParallel):
            flops, relu_count = model.module.get_total_count(max_latency=max_latency)
        else:
            flops, relu_count = model.get_total_count(max_latency=max_latency)
        # print(f'flops: {flops}, relu_count: {relu_count}')
        linear = self.hardware['linear'] * flops
        nonlinear = self.hardware['nonlinear'] * relu_count
        return linear, nonlinear
    
    def cal_coefficient_latency(self, model):
        if isinstance(model, torch.nn.DataParallel):
            flops_list, relu_count_list = model.module.get_flops_count_list()
        else:
            flops_list, relu_count_list = model.get_flops_count_list()
        # use y=x from 5 to 1
        coefficients_flops = [1 + (4 / (len(flops_list) - 1)) * i for i in range(len(flops_list))]
        coefficients_count = [10 - (4 / (len(flops_list) - 1)) * i for i in range(len(relu_count_list))]
        total_flops, total_count = 0, 0
        for i in range(len(flops_list)):
            co = coefficients_flops[i]
            w, flops = flops_list[i]
            temp_flops = 0
            for i, prob in enumerate(w):
                temp_flops += prob * flops[i]
            total_flops += temp_flops * co
        for i in range(len(relu_count_list)):
            co = coefficients_count[i]
            prob, count = relu_count_list[i]
            total_count += prob * count * co
        offline_linear = self.hardware['offline_linear'] * total_flops
        online_linear = self.hardware['online_linear'] * total_flops
        nonlinear = self.hardware['nonlinear'] * total_count
        return offline_linear + online_linear + nonlinear
    
    def _get_init_latency(self, model, upper):
        flops, relu_count = model.get_total_count(max_latency=True)
        print('linear cost: ', self.hardware['offline_linear'] * flops)
        print('nonlinear cost: ', self.hardware['nonlinear'] * relu_count)
        latency = self.hardware['nonlinear'] * relu_count * upper
        return latency