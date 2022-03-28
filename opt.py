#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
# optimizer set
import torch
import copy

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, model_size, factor=1, warmup=4000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        # self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def optimizer_select_and_init(model,
                              op_init_dict: dict,
                              sch_init_dict: dict) -> dict:
    """
    """
    temp_op_init_dict = copy.deepcopy(op_init_dict)
    type_optimizer = temp_op_init_dict['optimizer_name']
    temp_op_init_dict.pop('optimizer_name')
    # setup optimizer
    if type_optimizer == 'SGD':
        """
        op_init_dict for SGD
        op_init_dict = {
            'lr': float,
            'momentum': float,
            'weight_decay': float,
            'dampening': float,
            'nesterov': bool
        }
        """
        optimizer = torch.optim.SGD(model.parameters(),
                                    **temp_op_init_dict)
    elif type_optimizer == 'Adam':
        """
        op_init_dict for Adam
        op_init_dict = {
            'lr': float,
            'betas': Tuple[float, float],
            'eps': float,
            'weight_decay': float,
            'amsgrad': bool
        }
        """

        optimizer = torch.optim.Adam(model.parameters(),
                                     **temp_op_init_dict)
    else:
        raise ValueError('Optimizer Type Error.')

    temp_sch_init_dict = copy.deepcopy(sch_init_dict)
    type_scheduler = temp_sch_init_dict['scheduler name']
    temp_sch_init_dict.pop('scheduler name')
    # setup scheduler
    if type_scheduler == 'StepLR':
        """
        sch_init_dict for StepLR
        sch_init_dict = {
            'step_size': int,
            'gamma': float,
            'last_epoch': float,
            'verbose': bool
        }
        """
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    temp_sch_init_dict['step_size'],
                                                    gamma=temp_sch_init_dict['gamma'],
                                                    last_epoch=temp_sch_init_dict['last_epoch'],
                                                    verbose=temp_sch_init_dict['verbose'])
    elif type_scheduler == 'ExponentialLR':
        """
        sch_init_dict for Noam
        sch_init_dict = {
            'gamma': float, 
            'last_epoch': int, 
            'verbose': bool
        }
        """
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           temp_sch_init_dict['gamma'],
                                                           last_epoch=temp_sch_init_dict['last_epoch'],
                                                           verbose=temp_sch_init_dict['verbose'])
    elif type_scheduler == 'Noam':
        """
        sch_init_dict for Noam
        sch_init_dict = {
            'model_size': int, 
            'factor': float, 
            'warmup': int
        }
        """
        scheduler = NoamOpt(optimizer, **temp_sch_init_dict)
    elif type_scheduler == 'CyclicLR':
        """
        sch_init_dict = {
            'optimizer': (Optimizer) – Wrapped optimizer.
            'base_lr': (float or list)
            'max_lr': (float or list) 
            'step_size_up': (int) – Default: 2000
            'step_size_down': (int) –  Default: None
            'mode': (str) – One of {triangular, triangular2, exp_range}. Default: ‘triangular’
            'gamma': (float) – Constant in ‘exp_range’ scaling function: gamma**(cycle iterations) Default: 1.0
            'scale_fn': (function) – Default: None
            'scale_mode': (str) – {‘cycle’, ‘iterations’}Default: ‘cycle’
            'cycle_momentum': (bool) – Default: True
            'base_momentum': (float or list) –  Default: 0.8
            'max_momentum': (float or list) – Default: 0.9
            'last_epoch': (int) – Default: -1
            'verbose': (bool) – If True, prints a message to stdout for each update. Default: False.
        }
        """
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **temp_sch_init_dict)
    elif type_scheduler == 'OneCycleLR':
        """
        sch_init_dict = {
            'optimizer': (Optimizer) – Wrapped optimizer.
            'max_lr': (float or list) – Upper learning rate boundaries in the cycle for each parameter group.
            'total_steps': (int) – Default: None
            'epochs': (int) – Default: None
            'steps_per_epoch': (int) – Default: None
            'pct_start (float)': – Default: 0.3
            'anneal_strategy': (str) – {‘cos’, ‘linear’}, Default: ‘cos’
            'cycle_momentum': (bool) – Default: True
            'base_momentum': (float or list) – Default: 0.85
            'max_momentum': (float or list) – Default: 0.95
            'div_factor': (float) – Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25
            'final_div_factor': (float) – Default: 1e4
            'three_phase': (bool) – 
            'last_epoch': (int) – Default: -1
            'verbose': (bool) – If True, prints a message to stdout for each update. Default: False.
        }
        """
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **temp_sch_init_dict)
    elif type_scheduler == 'CosineAnnealingWarmRestarts':
        """
        sch_init_dict = {
            'T_0': (int) – Number of iterations for the first restart.
            'T_mult': (int, optional) – A factor increases 
            'eta_min': (float, optional) – Minimum learning rate. Default: 0.
            'last_epoch': (int, optional) – The index of last epoch. Default: -1.
            'verbose': (bool) – If True, prints a message to stdout for each update. Default: False.
        }
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **temp_sch_init_dict)
    elif type_scheduler == 'CosineAnnealingLR':
        """
        sch_init_dict = {
            'T_max': int – Maximum number of iterations.
            'eta_min': float – Minimum learning rate. Default: 0.
            'last_epoch': int – The index of last epoch. Default: -1.
            'verbose': bool – If True, prints a message to stdout for each update. Default: False
        }
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **temp_sch_init_dict)
    elif type_scheduler == 'None':
        scheduler = None
    else:
        raise ValueError('Scheduler Type Error.')

    # form into dict
    op_dict = {
        'type_optimizer': type_optimizer,
        'type_scheduler': type_scheduler,
        'op_init_dict': temp_op_init_dict,
        'sch_init_dict': temp_sch_init_dict,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    return op_dict


if __name__ == '__main__':
    import sys
    import torch.nn as nn

    optimizer_para = {
        'optimizer_name': 'Adam',
        'lr': 0.0001,
        'betas': (0.9, 0.98),
        'eps': 1e-9,
        'weight_decay': 0,
        'amsgrad': True,
    }
    scheduler_type_list = ['StepLR',
                           'ExponentialLR',
                           'Noam',
                           'CosineAnnealingLR',
                           'CosineAnnealingWarmRestarts',
                           'CyclicLR-triangular2',
                           'OneCycleLR-cos'
                           ]

    for scheduler_type in scheduler_type_list:
        if scheduler_type == 'StepLR':
            scheduler_para = {
                'scheduler name': 'StepLR',
                'step_size': 10,
                'gamma': 0.95,
                'last_epoch': -1,
                'verbose': False
            }
        elif scheduler_type == 'ExponentialLR':
            scheduler_para = {
                'scheduler name': 'ExponentialLR',
                'gamma': 0.98,
                'last_epoch': -1,
                'verbose': False
            }
        elif scheduler_type == 'Noam':
            scheduler_para = {
                'scheduler name': 'Noam',
                'model_size': 256,
                'factor': 2,
                'warmup': 96
            }
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler_para = {
                'scheduler name': 'CosineAnnealingLR',
                'T_max': 16,
                'eta_min': 0.0,
                'last_epoch': -1,
                'verbose': False
            }
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler_para = {
                'scheduler name': 'CosineAnnealingWarmRestarts',
                'T_0': 16,
                'T_mult': 1,
                'eta_min': 0.0,
                'last_epoch': -1,
                'verbose': False
            }
        elif scheduler_type == 'CyclicLR-triangular2':
            scheduler_para = {
                'scheduler name': 'CyclicLR',
                'base_lr': 0.001,
                'max_lr': 0.1,
                'step_size_up': 2000,
                'step_size_down':  None,
                'mode': 'triangular2',
                'gamma': 1.0,
                'scale_fn': None,
                'scale_mode': 'cycle',
                'cycle_momentum': False,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'last_epoch': -1,
                'verbose': False
            }
        elif scheduler_type == 'OneCycleLR-cos':
            scheduler_para = {
                'scheduler name': 'OneCycleLR',
                'max_lr': 0.1,
                'total_steps': None,
                'epochs': 10,
                'steps_per_epoch': 100,
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'cycle_momentum': False,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 25,
                'final_div_factor': 1e4,
                'three_phase': False,
                'last_epoch': -1,
                'verbose': False
            }
        else:
            raise ValueError('Scheduler Type Error.')
        op_sc_dict = optimizer_select_and_init(nn.Linear(100, 200), optimizer_para, scheduler_para)
    sys.exit(0)