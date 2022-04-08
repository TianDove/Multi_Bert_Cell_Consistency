#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import random
import torch
import utility_function
import model_define


def cls_label_to_one_hot(label,
                         num_classes: int):
    """"""
    one_hot = torch.zeros((label.shape[0], num_classes),
                          dtype=torch.float32)
    for b_id, lb in enumerate(iter(label)):
        one_hot[b_id, lb] = 1.0
    return one_hot


class MultiBertProcessing():
    """"""
    def __init__(self,
                 token_tuple: tuple,
                 rnd_para_data_path: str,
                 num_classes: int):
        """"""
        self.batch_size = None
        self.num_classes = num_classes
        self.rnd_para_dict = utility_function.read_pickle_file(rnd_para_data_path)
        for para_name, para_table in iter(self.rnd_para_dict.items()):
            self.rnd_para_dict[para_name] = torch.from_numpy(para_table).to(dtype=torch.float32)

        self.tokenizer = utility_function.Tokenizer(token_tuple)

    def pro(self, train_data, device):
        """"""
        self.batch_size = train_data['label'].shape[0]
        temp_label_arr = copy.deepcopy(train_data['label'])
        temp_label_arr = cls_label_to_one_hot(temp_label_arr,
                                              num_classes=self.num_classes)

        # delete label
        del train_data['label']

        m_out_data, nsp_lab = self.nsp_replace(train_data)

        nsp_lab_list = []
        for key, value in iter(nsp_lab.items()):
            nsp_lab_list.append(value[-1])
        nsp_lab_t = torch.tensor(nsp_lab_list, dtype=torch.long)
        nsp_one_hot = cls_label_to_one_hot(nsp_lab_t,
                                           num_classes=2).repeat(self.batch_size, 1)

        temp_token_dict = self.tokenize(m_out_data)
        temp_data_dict = utility_function.tensor_dict_to_device(temp_token_dict, device)

        return [temp_data_dict, temp_label_arr.to(device), nsp_one_hot.to(device)]

    def nsp_replace(self,
                    in_data,
                    max_rpl=1,
                    nsp_rate=0.5,
                    visulize=True):
        """"""
        rpl_num = 0
        nsp_positions_and_labels_dict = {}
        key_list = list(in_data.keys())
        key_back = copy.deepcopy(key_list)

        key_list.remove('ch1v')
        rnd_key = random.choice(key_list)
        rnd_para_data = None

        if random.random() < nsp_rate:

            # get para key
            key_back.remove(rnd_key)
            key_back.remove('ch1v')

            # get para for replace
            rnd_para_name = random.choice(key_back)
            rnd_choice_data = random.choice(self.rnd_para_dict[rnd_para_name])
            if len(rnd_choice_data.shape) < 2:
                rnd_choice_data = rnd_choice_data.reshape(1, -1)
            rnd_para_data = rnd_choice_data.repeat(self.batch_size, 1)
            in_data[rnd_key] = rnd_para_data

            # 0 - not next para, 1 - next para
            # [true para pos, sub para name, 0/1]
            temp_nsp_label = [rnd_key, rnd_para_name, 0]
        else:
            rnd_para_name = rnd_key
            temp_nsp_label = [rnd_key, rnd_para_name, 1]

        nsp_positions_and_labels_dict[rnd_key] = temp_nsp_label
        rpl_num += 1

        return in_data, nsp_positions_and_labels_dict

    def tokenize(self,
                 in_data):
        """"""
        # tokenize in_data
        temp_token_dict = {}
        temp_label_arr = []
        for key, value in iter(in_data.items()):
            # B * L
            temp_para_arr = value
            temp_token_list = []
            for row in iter(temp_para_arr):
                # 1 * L
                temp_token_arr, _ = self.tokenizer.tokenize(row.unsqueeze(0))
                # update token list
                temp_token_list.append(temp_token_arr.unsqueeze(0))

            # update token dict
            # B * n_token * token_len
            temp_token_dict[key] = torch.cat(temp_token_list, dim=0)

        return temp_token_dict


class BaseProcessing():
    """"""
    def __init__(self,
                 num_classes: int):
        """"""
        self.num_classes = num_classes
        self.total_len = None
        self.batch_size = None

    def pro(self,
            train_data: torch.Tensor,
            device: torch.device):
        """"""
        self.batch_size = train_data['label'].shape[0]
        temp_label_arr = copy.deepcopy(train_data['label']).to(device)
        temp_label_arr = cls_label_to_one_hot(temp_label_arr,
                                              num_classes=self.num_classes)

        # delete label
        del train_data['label']

        temp_para_list = []
        for key, value in iter(train_data.items()):
            temp_para_list.append(value)
        temp_para_tensor = torch.cat(temp_para_list, dim=1)

        self.total_len = temp_para_tensor.shape[1]

        return [temp_para_tensor.to(device),
                temp_label_arr.to(device)]


