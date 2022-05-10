#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os

import torch
import torch.nn as nn
import model_define

class CAEDecoder(nn.Module):
    def __init__(self,
                 conv_out_ch,
                 flatten_sz,
                 unflatten_sz,
                 hid,
                 conv_k_sz=3,
                 pool_k_sz=2,
                 dropout: float = 0.1):
        """"""
        super(CAEDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features=hid, out_features=flatten_sz // 2)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(in_features=flatten_sz // 2, out_features=flatten_sz,)
        self.unflat = nn.Unflatten(-1, unflatten_sz)
        self.conT1 = CAEDecoderLayer(conv_out_ch[2], conv_out_ch[1])
        self.conT2 = CAEDecoderLayer(conv_out_ch[1], conv_out_ch[0])
        self.conT3 = CAEDecoderLayer(conv_out_ch[0], 1)


    def forward(self, x, idx, sz):
        """"""
        res = self.linear1(x)
        res = self.act(res)
        res = self.linear2(res)
        res = self.unflat(res)
        res = self.conT1(res, idx[-1], sz[-1])
        res = self.conT2(res, idx[-2], sz[-2])
        res = self.conT3(res, idx[-3], sz[-3])
        return res


class CAEEncoder(nn.Module):
    def __init__(self,
                 conv_out_ch,
                 flatten_sz,
                 hid,
                 conv_k_sz=3,
                 pool_k_sz=2,
                 dropout: float = 0.1):
        """"""
        super(CAEEncoder, self).__init__()

        self.idx_list = None
        self.data_sz_list = None

        self.conv1 = CAEEncoderLayer(1, conv_out_ch[0])
        self.conv2 = CAEEncoderLayer(conv_out_ch[0], conv_out_ch[1])
        self.conv3 = CAEEncoderLayer(conv_out_ch[1], conv_out_ch[2])
        self.flat = nn.Flatten()
        self.linear1 =  nn.Linear(in_features=flatten_sz, out_features=flatten_sz // 2)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(in_features=flatten_sz // 2, out_features=hid)

    def forward(self, x):
        """"""
        self.idx_list = []
        self.data_sz_list = []

        res = x

        res, idx1, sz1 = self.conv1(res)
        self.data_sz_list.append(sz1)
        self.idx_list.append(idx1)

        res, idx2, sz2 = self.conv2(res)

        self.data_sz_list.append(sz2)
        self.idx_list.append(idx2)

        res, idx3, sz3 = self.conv3(res)

        self.data_sz_list.append(sz3)
        self.idx_list.append(idx3)

        res = self.flat(res)
        res = self.linear1(res)
        res = self.act(res)
        res = self.linear2(res)

        return res, self.idx_list, self.data_sz_list

class CAEEncoderLayer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 conv_k_sz = 3,
                 pool_k_sz = 2,
                 dropout: float = 0.1):
        """"""
        super(CAEEncoderLayer, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=conv_k_sz)
        self.pool = nn.MaxPool1d(kernel_size=pool_k_sz, return_indices=True)
        self.act = nn.ReLU()

    def forward(self, x):
        """"""
        res = self.conv(x)

        sz = res.shape

        res, idx = self.pool(res)
        res = self.act(res)
        return res, idx, sz


class CAEDecoderLayer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 conv_k_sz=3,
                 pool_k_sz=2,
                 dropout: float = 0.1):
        """"""
        super(CAEDecoderLayer, self).__init__()

        self.conv_t = nn.ConvTranspose1d(in_channels=in_ch, out_channels=out_ch, kernel_size=conv_k_sz)
        self.pool = nn.MaxUnpool1d(kernel_size=pool_k_sz)
        self.act = nn.ReLU()

    def forward(self, x, idx, sz):
        """"""
        res = self.act(x)
        res = self.pool(res, idx, output_size=sz)
        res = self.conv_t(res)
        return res


class CAE(nn.Module):
    def __init__(self,
                 para: str,
                 in_dim: int,
                 conv_out_ch: tuple = (256, 128, 64),
                 conv_k_sz = 3,
                 pool_k_sz = 2,
                 hid: int = 10,
                 loss_func=None,
                 dropout: float = 0.1):
        """"""
        super(CAE, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

        self.in_dim = in_dim
        self.conv_out_ch = conv_out_ch
        self.conv_k_sz = conv_k_sz
        self.pool_k_sz = pool_k_sz
        self.hid = hid
        self.loss_func = loss_func
        self.dropout = dropout

        self.encoder_conv_out_sz = self.cal_encoder_con_out_sz()
        self.flatten_sz = self.encoder_conv_out_sz[-2] * self.encoder_conv_out_sz[-1]
        self.unflatten_sz = (self.encoder_conv_out_sz[-2], self.encoder_conv_out_sz[-1])

        self.encoder = CAEEncoder(self.conv_out_ch, self.flatten_sz, self.hid)
        self.decoder = CAEDecoder(self.conv_out_ch, self.flatten_sz, self.unflatten_sz, self.hid)

    def cal_encoder_con_out_sz(self):
        """"""
        tmp_in_len = self.in_dim
        tmp_in_ch = 1
        for out_ch in self.conv_out_ch:
            tmp_conv_out_len = tmp_in_len - self.conv_k_sz + 1
            temp_pool_out_len = math.floor((tmp_conv_out_len - self.pool_k_sz) / self.pool_k_sz) + 1
            # print(temp_pool_out_len)
            tmp_in_len = temp_pool_out_len

        return (1, self.conv_out_ch[-1], temp_pool_out_len)


    def forward(self, x, y=None, train_mode:str=None):
        """"""
        res, idx, sz = self.encoder(x)
        inter_res = res
        if train_mode == 'test':
            return (inter_res, y)

        res = self.decoder(inter_res, idx, sz)

        self.model_batch_out = res

        if (y is not None) and (self.loss_func is not None):
            ls = self.loss_func(res, y)
            self.model_batch_loss = ls
            return ls
        else:
            if y is not None:
                return (res, y)
            else:
                raise ValueError('Label Data is Empty or None.')


    def get_out(self):
        """"""
        return self.model_batch_out


    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_save_model(self):
        """"""
        attr_need = self.__dict__
        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        return model


class BaseLine_DDDOM(nn.Module):
    def __init__(self,
                 cae_model_path,
                 num_cls: int = 8,
                 num_paras: int = 5,
                 cae_hid: int = 10,
                 loss_func=None,
                 dropout: float = 0.1):
        """"""
        super(BaseLine_DDDOM, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

        self.cae_model_path = cae_model_path
        self.num_cls = num_cls
        self.num_paras = num_paras
        self.cae_hid = cae_hid
        self.loss_func = loss_func
        self.dropout = dropout


        self.cae_dict = {
            'ch1v': CAE('ch', 256),
            'ch2v': CAE('ch', 256),
            'dcv': CAE('ch', 256),
            'ch3v': CAE('ch', 256),
            'ch3c': CAE('ch', 256),
        }

        if os.path.exists(self.cae_model_path):
            self.load_multi_cae()
        else:
            raise FileNotFoundError('CAE Model Path Not Exist.')

        self.mlp = nn.Sequential(
            model_define.Dense_Sigmoid(self.cae_hid * self.num_paras, 128, dropout=self.dropout),
            model_define.Dense_Sigmoid(128, 64, dropout=self.dropout),
            model_define.Dense_Sigmoid(64, 32, dropout=self.dropout),
            nn.Linear(in_features=32, out_features=self.num_cls),
            nn.Softmax(dim=-1),
        )

    def load_multi_cae(self):
        """"""
        cae_model_list = os.listdir(self.cae_model_path)

        for cae_file_name in cae_model_list:
            temp_cae_model_path = os.path.join(self.cae_model_path, cae_file_name)

            if os.path.exists(temp_cae_model_path):
                temp_model_dict = torch.load(temp_cae_model_path)
                temp_model = temp_model_dict['model']
            else:
                raise FileNotFoundError(f'File No Find: {temp_cae_model_path}')

            temp_file_name, temp_file_type = os.path.splitext(cae_file_name)
            temp_file_name_list = temp_file_name.split('_')
            temp_model_para = temp_file_name_list[-1]

            self.cae_dict[temp_model_para].set_model_attr(temp_model)
            for (name, param) in self.cae_dict[temp_model_para].named_parameters():
                param.requires_grad = False


    def forward(self, x, y=None, train_mode:str=None):
        """"""

        res = x

        para_hid = []
        for key, temp_cae in self.cae_dict.items():
            temp_para_data = x[key]
            temp_hid, _ = temp_cae(temp_para_data, y, train_mode='test')
            para_hid.append(temp_hid)
        para_hid_tensor = torch.cat(para_hid, dim=1)

        res = self.mlp(para_hid_tensor)


        self.model_batch_out = res

        if (y is not None) and (self.loss_func is not None):
            ls = self.loss_func(res, y)
            self.model_batch_loss = ls
            return ls
        else:
            if y is not None:
                return (res, y)
            else:
                raise ValueError('Label Data is Empty or None.')


    def get_out(self):
        """"""
        return self.model_batch_out


    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_save_model(self):
        """"""
        attr_need = self.__dict__
        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        return model