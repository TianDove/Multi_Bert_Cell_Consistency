#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import random
import torch
import torch.nn.functional as F
import utility_function
import model_define


def cls_label_to_one_hot(label,
                         num_classes: int):
    """"""
    one_hot = torch.zeros((label.shape[0], num_classes),
                          dtype=torch.float32)
    for b_id, lb in enumerate(iter(label)):
        lb = lb.to(torch.long).item()
        one_hot[b_id, lb] = 1.0
    return one_hot


class BaseProcessing():
    """"""
    def __init__(self,
                 num_classes: int):
        """"""
        self.num_classes = num_classes
        self.total_len = None
        self.batch_size = None
        self.batch_size_flag = False
        self.print_info_flag = False

    def pro(self,
            train_data: dict,
            train_mode: str,
            device: torch.device):
        """"""
        if self.batch_size_flag is False:
            self.batch_size = train_data['label'].shape[0]
            self.batch_size_flag = True
        temp_label_arr = torch.clone(train_data['label']).to(torch.long)
        temp_label_arr = F.one_hot(temp_label_arr, self.num_classes)

        # delete label
        del train_data['label']

        temp_para_list = []
        for key, value in iter(train_data.items()):
            temp_para_list.append(value)
        temp_para_tensor = torch.cat(temp_para_list, dim=1)

        self.total_len = temp_para_tensor.shape[1]

        if not self.print_info_flag:
            self.print_info_flag = True
            print('-' * 64)
            print(f'PreProcess Type: {self.__class__}')
            print(f'Num Class: {self.num_classes}')
            print(f'| Para | (batch, Len) |')
            for key, val in train_data.items():
                print(f'| {key} | {(val.shape)} |')
            print(f'Total Len: {self.total_len}')
            print('-' * 64)

        return [temp_para_tensor.to(device),
                temp_label_arr.to(device)]


class MyMultiBertModelProcessing(object):
    """"""
    def __init__(self,
                 num_classes,
                 token_tuple: tuple,
                 rnd_para_dict: dict,
                 params: tuple = None):
        """"""
        self.token_tuple = token_tuple
        self.num_classes = num_classes
        self.batch_size_flag = False
        self.batch_size = None
        self.rnd_para_dict = rnd_para_dict
        self.num_token = None
        self.t_len = token_tuple[0]
        self.overlap = token_tuple[1]
        self.step = token_tuple[2]
        self.params = params

    def pro(self,
            train_data: dict,
            train_mode: str,
            device: torch.device) ->list:
        """"""
        if self.batch_size_flag is False:
            self.batch_size = train_data['label'].shape[0]
            self.batch_size_flag = True

        temp_label_one_hot = None
        temp_label_tensor = torch.clone(train_data['label'])
        temp_label_long_tensor = temp_label_tensor.to(torch.long)
        temp_label_long_tensor_device = temp_label_long_tensor.to(device)
        temp_label_one_hot = F.one_hot(temp_label_long_tensor_device, self.num_classes)

        # delete label
        del train_data['label']
        rpl_label_onehot_tensor_device = None
        temp_data_dict = None

        if self.params:
            temp_data_dict = {}
            for key in self.params:
                temp_data_dict[key] = train_data[key]
        else:
            temp_data_dict = train_data


        if train_mode == 'pretrain':
            rpl_label_onehot_tensor_device = []
            temp_data_dict, rpl_label_tensor = self.next_para_replace(temp_data_dict)
            rpl_label_onehot_tensor = F.one_hot(rpl_label_tensor, 2)
            rpl_label_onehot_tensor_cp = torch.clone(rpl_label_onehot_tensor)
            rpl_label_onehot_tensor_device = rpl_label_onehot_tensor_cp.to(device)

        temp_data_token_dict = self.tokenize_dict(temp_data_dict)

        temp_out_data_dict = None
        temp_out_data_dict = utility_function.tensor_dict_to_device(temp_data_token_dict, device)

        return (temp_out_data_dict, temp_label_one_hot, rpl_label_onehot_tensor_device, train_mode)

    def cal_num_in_token(self):
        """"""
        para_len = (161, 177, 247, 218, 218)
        in_len = self.token_tuple[0]

        num_in_token = 0
        for t_len in para_len:
            temp_tensor = torch.ones((1, t_len), dtype=torch.float32)
            temp_tokens = self.tokenize_tensor(temp_tensor)
            temp_num_token = temp_tokens.shape[1]
            num_in_token = num_in_token + temp_num_token

        return num_in_token

    def next_para_replace(self,
                          batch_data: dict,
                          rpl_rate: float = 0.5):
        """"""

        tmp_out_tensor_dict = {}
        for key, val in batch_data.items():
            tmp_out_tensor_dict[key] = torch.clone(val)

        para_type = list(tmp_out_tensor_dict.keys())
        batch_size = tmp_out_tensor_dict[para_type[0]].shape[0]

        # dont replace first para
        para_type.remove(para_type[0])

        # select a para to replace
        gt_key = random.choice(para_type)

        # random replace para
        if random.random() < rpl_rate:
            para_type_bak = copy.deepcopy(para_type)
            # use other para to replace gt_key para
            para_type_bak.remove(gt_key)

            # get para to replace
            rlp_key = random.choice(para_type_bak)
            rnd_choice_data = random.choice(self.rnd_para_dict[rlp_key])
            rnd_choice_data_cp = torch.clone(rnd_choice_data)

            rnd_choice_data_uq = rnd_choice_data_cp.unsqueeze(0)  # (Batch_size, Data_size)
            rnd_para_data_rep = rnd_choice_data_uq.repeat(batch_size, 1)
            tmp_out_tensor_dict[gt_key] = rnd_para_data_rep
            replace_abel = torch.zeros(batch_size, dtype=torch.long)   # not next
        else:
            rlp_key = gt_key
            replace_abel = torch.ones(batch_size, dtype=torch.long)  # next

        #label one-hot
        replace_abel_tensor = torch.clone(replace_abel)

        # visualize
        (gt_key, rlp_key)
        return tmp_out_tensor_dict, replace_abel_tensor

    def tokenize_dict(self,
                 batch_data: dict) -> dict:
        """"""
        num_token = 0

        temp_token_dict = {}
        for key, val in batch_data.items():
            temp_tokens = self.tokenize_tensor(val)
            temp_token_dict[key] = torch.clone(temp_tokens)

        return temp_token_dict

    def tokenize_tensor(self, data_tensor: torch.tensor) -> torch.tensor:
        """
        (Batch, Date)
        """
        temp_token_tensor = None
        d_size = data_tensor.shape
        in_temp = data_tensor
        if d_size[-1] > self.t_len:
            r_mod = d_size[1] % self.t_len
            if not self.overlap:
                if r_mod != 0:
                    pad_num = 0
                    num_of_padding = self.t_len - r_mod
                    pad_arr = torch.ones(num_of_padding) * pad_num
                    pad_arr = pad_arr.repeat(d_size[0], 1)
                    in_temp = torch.cat([data_tensor, pad_arr], dim=-1)
                out_data = in_temp.reshape(d_size[0], -1, self.t_len)
                self.num_token = out_data.shape[1]
                return out_data
            else:
                raise ValueError('Non OverLap olny.')



    def mask_token_replace(self):
        """"""
        pass


class CAEProcessing():
    """"""
    def __init__(self,
                 train_mode,
                 para):
        """"""
        self.train_mode = train_mode
        self.para = para
        self.batch_size_flag = False
        self.print_info_flag = False

        self.batch_size = None

    def pro(self,
            train_data: dict,
            train_mode,
            device: torch.device):
        """"""
        temp_data = train_data[self.para]

        temp_data_unsq = temp_data.unsqueeze(1)
        temp_label = torch.clone(temp_data_unsq)

        return [temp_data_unsq.to(device),
                temp_label.to(device),
                train_mode]


class DDDOMProcessing():
    """"""
    def __init__(self,
                 train_mode,
                 num_classes):
        """"""
        self.train_mode = train_mode
        self.num_classes = num_classes
        self.batch_size_flag = False
        self.print_info_flag = False

        self.batch_size = None

    def pro(self,
            train_data: dict,
            train_mode,
            device: torch.device):
        """"""
        temp_label_one_hot = None
        temp_label_tensor = torch.clone(train_data['label'])
        temp_label_long_tensor = temp_label_tensor.to(torch.long)
        temp_label_long_tensor_device = temp_label_long_tensor.to(device)
        temp_label_one_hot = F.one_hot(temp_label_long_tensor_device, self.num_classes)

        # delete label
        del train_data['label']

        temp_data = {}
        for key, val in train_data.items():
            temp_val_unsq = val.unsqueeze(1)
            temp_val_unsq_dev = temp_val_unsq.to(device)
            temp_data[key] = torch.clone(temp_val_unsq_dev)


        return [temp_data,
                temp_label_one_hot.to(device),
                train_mode]


if __name__ == '__main__':
    m_prepro = MyMultiBertModelProcessing(8, (48, False, 1), None)
    m_prepro.cal_num_in_token()