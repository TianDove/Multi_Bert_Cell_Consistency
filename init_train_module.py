#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.data import Dataset, DataLoader

import opt
import utility_function
import dataset_and_dataloader


def init_device(device_name: str, gpu_idx=None):
    """"""
    tmp_device = None
    if device_name == 'cpu':
        tmp_device = torch.device('cpu')
    elif device_name == 'gpu':
        if gpu_idx is not None:
            tmp_device = utility_function.try_gpu(gpu_idx)
        else:
            tmp_device = utility_function.try_all_gpus()

    return tmp_device


def rnd_token_loader(rnd_token_file_path: str, device: torch.device = None) -> torch.tensor:
    """"""
    if os.path.exists(rnd_token_file_path):
        temp_rnd_token_array = utility_function.read_pickle_file(rnd_token_file_path)
        temp_rnd_token_tensor = torch.from_numpy(temp_rnd_token_array).to(dtype=torch.float32)
        if device is not None:
            temp_rnd_token_tensor = temp_rnd_token_tensor.to(device)
        return temp_rnd_token_tensor
    else:
        raise FileNotFoundError


def rnd_para_loader(rnd_para_file_path: str, device: torch.device = None) -> dict[str, torch.tensor]:
    """"""
    if os.path.exists(rnd_para_file_path):
        temp_rnd_para_tensor_dict = {}
        temp_rnd_para_dict = utility_function.read_pickle_file(rnd_para_file_path)
        for para_name, para_val in temp_rnd_para_dict.items():
            temp_para_val = torch.from_numpy(para_val).to(dtype=torch.float32)
            if device is not None:
                temp_para_val = temp_para_val.to(device)

            temp_rnd_para_tensor_dict[para_name] = temp_para_val
        return temp_rnd_para_tensor_dict
    else:
        raise FileNotFoundError


def init_data_loader_dict(data_set_file_path: str,
                          train_mode: str,
                          batch_sz: list[int],
                          is_shuffle: bool = True,
                          num_of_worker: int = 1) -> dict[str, DataLoader]:
    """"""
    if os.path.exists(data_set_file_path):
        temp_data_set_dict = utility_function.read_pickle_file(data_set_file_path)

        # decide whether use pretrian data set
        set_name_list = list(temp_data_set_dict.keys())
        if train_mode == 'pretrain':
            temp_ch_data_set_dict = {train_mode: temp_data_set_dict[train_mode]}
        elif train_mode in ['train', 'val', 'test', 'finetune']:
            temp_train_set = temp_data_set_dict['train']
            temp_pretrain_set = temp_data_set_dict['pretrain']
            temp_pretrain_cat_train_dict = {**temp_pretrain_set, **temp_train_set}
            del temp_data_set_dict['pretrain']
            temp_data_set_dict['train'] = temp_pretrain_cat_train_dict
            temp_ch_data_set_dict = temp_data_set_dict
        else:
            raise ValueError('Train Mode Error.')

        temp_data_loader_dict = {}
        num_data_set = len(temp_ch_data_set_dict)
        if len(batch_sz) != num_data_set:
            raise ValueError('Num of Data set != len(batch_sz)')

        for bsz_idx, data_set_name in enumerate(temp_ch_data_set_dict):
            temp_data_set_dict = temp_ch_data_set_dict[data_set_name]
            temp_data_set_loader = dataset_and_dataloader.CellDataLoader. \
                creat_data_loader(data_set_dict=temp_data_set_dict,
                                  batch_sz=batch_sz[bsz_idx],
                                  is_shuffle=is_shuffle,
                                  num_of_worker=num_of_worker)
            temp_data_loader_dict[data_set_name] = temp_data_set_loader

        return temp_data_loader_dict

    else:
        raise FileNotFoundError


def init_model(model: torch.nn.Module,
               model_param_dict: dict,
               device: torch.device = None) -> torch.nn.Module:
    """"""
    if model != {}:
        temp_model = model(**model_param_dict)
        temp_model = utility_function.init_model(temp_model)
        temp_model = temp_model.to(device)
        return temp_model
    else:
        raise ValueError('Model Parameter Dict Empty.')


def init_optimaizer_scheduler(inited_model: torch.nn.Module,
                              opt_param_dict: dict,
                              shc_param_dict: dict = None):
    """"""
    if (opt_param_dict != {}) and (shc_param_dict != {}):
        op_sch_dict = opt.optimizer_select_and_init(inited_model,
                                                    opt_param_dict,
                                                    shc_param_dict)
        return op_sch_dict['optimizer'], op_sch_dict['scheduler']
    else:
        raise ValueError('Optimizer Parameter Dict or Scheduler Parameter Dict Empty.')


class ModelRun(object):
    """"""
    def __init__(self):
        """"""
        pass

    def run(self):
        """"""
        pass
