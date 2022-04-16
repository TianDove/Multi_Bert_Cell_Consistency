#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import copy

import pandas as pd
import torch
import torch.nn as nn
# from torchviz import make_dot

import utility_function



class MyDownStreamHead(nn.Module):
    """"""
    def __init__(self,
                 num_class,
                 num_token,
                 embedding_token_dim,
                 dropout=0.1):
        """"""
        super(MyDownStreamHead, self).__init__()
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=num_token * embedding_token_dim, out_features=num_class)
        self.activ = nn.GELU()
        self.norm = nn.BatchNorm1d(num_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, in_data):
        """

        :param in_data: (Batch_size, num_token, embedding dim)
        :return:(Batch_size, num_Class)
        """
        in_data = self.flat(in_data)
        in_data = self.dropout(in_data)
        in_data = self.linear(in_data)
        in_data = self.activ(in_data)
        in_data = self.norm(in_data)
        in_data = self.softmax(in_data)
        return in_data


class NextParaPred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, embedding_token_dim, dropout=0.1):
        super(NextParaPred, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(embedding_token_dim, embedding_token_dim // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embedding_token_dim // 2)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(embedding_token_dim // 2, 2)
        self.outlayer = nn.Softmax(dim=-1)

    def forward(self, in_data):
        # in_data: [batch_size, embedding_token_dim]
        out_data = self.dropout1(in_data)
        out_data = self.linear1(out_data)
        out_data = self.act(out_data)
        out_data = self.norm(out_data)
        out_data = self.dropout2(out_data)
        out_data = self.linear2(out_data)
        out_data = self.outlayer(out_data)
        return out_data


class MaskTokenPred(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, in_len, embedding_token_dim, dropout=0.1):
        super(MaskTokenPred, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(embedding_token_dim, embedding_token_dim // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embedding_token_dim // 2)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(embedding_token_dim // 2, in_len)

    def forward(self, in_data):
        out_data = self.dropout1(in_data)
        out_data = self.linear1(out_data)
        out_data = self.act(out_data)
        out_data = self.norm(out_data)
        out_data = self.dropout2(out_data)
        out_data = self.linear2(out_data)
        return out_data


# define token embedding
class CovTokenEmbedding(nn.Module):
    """"""
    def __init__(self, max_len, dropout=0.1):
        """"""
        super(CovTokenEmbedding, self).__init__()
        self.in_ch = max_len
        self.stride = (1, 1, 1, 1, 1, 1)
        self.ksz = (7, 5, 3, 3, 2, 2)
        self.cov_blk = nn.Sequential()
        for i in range(len(self.stride)):
            self.cov_blk.add_module(f'conv{i}',
                                    nn.Conv1d(in_channels=self.in_ch,
                                              out_channels=self.in_ch,
                                              kernel_size=self.ksz[i],
                                              stride=self.stride[i],
                                              padding=0))

            self.cov_blk.add_module(f'drop{i}', nn.Dropout(dropout))

    def forward(self, in_data):
        """
        in_data: Batch * num_token * data_len
        :param in_data:
        :return:
        """
        # token embedding
        token_embedding_list = []
        for token_idx in range(in_data.shape[1]):
            temp_token_embedding = self.cov_blk(in_data[:, token_idx, :].unsqueeze(1))
            token_embedding_list.append(temp_token_embedding)
        out_data = torch.cat(token_embedding_list, dim=1)
        return out_data


class MyMultiBertModel(nn.Module):
    """"""
    def __init__(self,
                 device,
                 token_len: int,
                 rnd_token: torch.tensor,
                 max_num_seg: int,
                 max_num_token: int,
                 embedding_dim: int,
                 n_layer:int,
                 n_head: int,
                 n_hid: int,
                 dropout: float = 0.1):
        """"""
        super(MyMultiBertModel, self).__init__()
        # record input param
        self.device = device
        self.token_len = token_len
        self.rnd_token = rnd_token
        self.max_num_seg = max_num_seg
        self.max_num_token = max_num_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_hid = n_hid
        self.dropout = dropout

        # scalar
        self.scale = math.sqrt(embedding_dim)
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None
        self.NPP_batch_loss = None
        self.MTP_batch_loss = None
        self.num_token = None
        self.batch_size = None

        # model layer and operation
        # Special token embedding
        # 'PAD': 0,
        # 'SOS': 1,
        # 'EOS': 2,
        # 'STP': 3,
        # 'CLS': 4,
        # 'MASK': 5
        self.special_token_embedding = nn.Embedding(num_embeddings=6,
                                                    embedding_dim=self.token_len,
                                                    padding_idx=0,
                                                    max_norm=3)

        # Token Embedding
        self.token_embedding = CovTokenEmbedding(1, dropout=dropout)

        # Segment Embedding
        self.segment_embedding = nn.Embedding(num_embeddings=max_num_seg,
                                              embedding_dim=embedding_dim,
                                              max_norm=3)

        # Positional Embedding
        self.position_embedding = nn.Parameter(torch.randn(1, max_num_token, embedding_dim))  # (0, 1)

        # Transformer Encoder
        self.encoder_blk = nn.Sequential()
        for i in range(n_layer):
            self.encoder_blk.add_module(f'encoder{i}', nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                  nhead=n_head,
                                                                                  dim_feedforward=n_hid,
                                                                                  dropout=self.dropout,
                                                                                  activation='gelu',
                                                                                  batch_first=True))

        # Mask Token Prediction Head
        self.mask_token_pre_head = MaskTokenPred(in_len=self.token_len,
                                                 embedding_token_dim=embedding_dim,
                                                 dropout=dropout)
        self.mask_token_pred_loss = nn.MSELoss()

        # Next Parameter Prediction Head
        self.next_para_pre_head =  NextParaPred(embedding_token_dim=embedding_dim,
                                                dropout=dropout)

        self.next_para_pre_loss = nn.CrossEntropyLoss()

        # DownStream Head
        self.downstream_head = MyDownStreamHead(num_class=8,
                                                num_token= 41,
                                                embedding_token_dim=embedding_dim,
                                                dropout=dropout)

        self.downstream_loss = nn.CrossEntropyLoss()

    def forward(self,
                inputs: dict,
                label: torch.tensor = None,
                rpl_label: torch.tensor = None,
                train_mode: str = 'pretrain',
                model_dict=None):
        """"""
        with torch.autograd.set_detect_anomaly(True):
            # with torch.autograd.set_detect_anomaly(True):
            inter_res = inputs
            mlm_label = None
            self.get_batch_size(inter_res)

            # special token tensor
            temp_sp_idx_tensor = torch.tensor([0, 1, 2, 3, 4, 5],
                                              dtype=torch.long,
                                              device=self.device)
            sp_token_tensor = self.special_token_embedding(temp_sp_idx_tensor)
            sp_token_tensor = sp_token_tensor.unsqueeze(0)
            sp_token_tensor = sp_token_tensor.repeat(self.batch_size, 1, 1)

            if train_mode == 'pretrain':
                inter_res, mlm_label = self.mask_token_batch(sp_token_tensor, inter_res)

            inter_res, segment_index, mlm_label = self.token_insert_batch(sp_token_tensor, inter_res, mlm_label)
            inter_res = self.token_embedding(inter_res)

            # get segments embedding
            temp_seg_idx_tensor = torch.tensor(segment_index,
                                               dtype=torch.long,
                                               device=self.device)
            segments_embedding = self.segment_embedding(temp_seg_idx_tensor)

            seg_pos_embedding = segments_embedding + self.position_embedding[:, 0:len(segment_index), :]

            seg_pos_embedding = seg_pos_embedding.repeat(self.batch_size, 1, 1)

            # add position embedding and segment embedding to the data
            inter_res = inter_res + (self.scale * seg_pos_embedding)

            # encoder input B * L * EB
            encoder_out = self.encoder_blk(inter_res)


            if train_mode == 'pretrain':
                mlm_pred = None
                if mlm_label != []:
                    temp_token_for_pred = []
                    for idx in mlm_label:
                        pos = idx[0]
                        temp_token = encoder_out[:, pos, :].unsqueeze(1)
                        temp_token_for_pred.append(temp_token)
                    temp_token_for_pred = torch.cat(temp_token_for_pred, dim=1)

                    mlm_pred = self.mask_token_pre_head(temp_token_for_pred)

                    # cat label
                    mlm_label_gt_list = []
                    for id, gt in mlm_label:
                        temp_gt = gt.unsqueeze(1)
                        mlm_label_gt_list.append(temp_gt)
                    mlm_label_tensor = torch.cat(mlm_label_gt_list, dim=1)

                    self.MTP_batch_loss = self.mask_token_pred_loss(mlm_pred, mlm_label_tensor)

                # get 'CLS' token in batch
                if rpl_label is None:
                    raise ValueError('Error:Replace Parameter Label is None, When in Pre-Train Mode.')

                batch_cls_token = encoder_out[:, 0, :]
                nsp_pred = self.next_para_pre_head(batch_cls_token)
                self.NPP_batch_loss = self.next_para_pre_loss(nsp_pred, rpl_label)

                if self.MTP_batch_loss is None:
                    self.MTP_batch_loss = 0.0
                    self.model_batch_loss = self.NPP_batch_loss
                else:
                    self.model_batch_loss = self.MTP_batch_loss + self.NPP_batch_loss

            else:
                self.model_batch_out = self.downstream_head(encoder_out)
                self.model_batch_loss = self.downstream_loss(self.model_batch_out, label)

        return self.model_batch_loss

    def get_segment_embedding(self, segment_index: list):
        """"""
        if segment_index != []:
            temp_seg_idx_tensor = torch.tensor(segment_index,
                                               dtype=torch.long,
                                               device=self.device)
            seg_embed_tensor = self.segment_embedding(temp_seg_idx_tensor)
            return seg_embed_tensor

        else:
            raise ValueError('Empty Segment Index List')

    def mask_token_batch(self,
                         sp_token_tensor,
                         batch_data: dict,
                         mask_rate=0.15):
        """"""
        batch_data_size = self.batch_size
        mlm_para_label_list = []
        # get mask token embedding

        # ##################
        # data_df_list = []
        # ##################

        out_data_dict = {}
        in_mask_token = torch.clone(sp_token_tensor[:, 5, :])
        for para_name, para_value in iter(batch_data.items()):
            mlm_token_label_list = []
            temp_para_value = para_value
            # ###########################################################################
            # data_df = pd.DataFrame(data=para_value[0, :, :].cpu().detach().numpy())
            # ###########################################################################
            rpl_count = 0
            for token_idx in range(temp_para_value.shape[1]):
                temp_org_token = temp_para_value[:, token_idx, :]
                temp_pred_positions_and_labels = [token_idx, temp_org_token, 'None']
                if random.random() < mask_rate:
                    rpl_count = rpl_count + 1

                    if random.random() < 0.8:
                        # 80%的时间：将词替换为“<mask>”词元
                        temp_mask_token = in_mask_token
                        mask_type = 'mask'
                    else:
                        if random.random() < 0.5:
                            # 10%的时间：保持词不变
                            temp_mask_token = temp_org_token
                            mask_type = 'org'
                        else:
                            # 10%的时间：用随机词替换该词
                            temp_mask_token = random.choice(self.rnd_token)
                            temp_mask_token = copy.deepcopy(temp_mask_token)
                            temp_mask_token = temp_mask_token.to(self.device)
                            temp_mask_token = temp_mask_token.unsqueeze(0)
                            temp_mask_token = temp_mask_token.repeat(self.batch_size, 1)
                            mask_type = 'rnd'

                    # replace token in the sequence
                    temp_para_value[:, token_idx, :] = torch.clone(temp_mask_token)
                    temp_pred_positions_and_labels[-1] = mask_type

                    # #############################################################################################
                    # data_df.iloc[token_idx, :] = pd.Series(data=temp_mask_token[0, :].cpu().detach().numpy())
                    # data_df = data_df.rename(index={token_idx: mask_type})
                    # #############################################################################################
                    # token scale label list
                    mlm_token_label_list.append(temp_pred_positions_and_labels)

            out_data_dict[para_name] = temp_para_value
            # para scale label list
            mlm_para_label_list.append(mlm_token_label_list)
            
            # #############################
            # data_df_list.append(data_df)
            # #############################
        
        # ###############################################
        # all_data_df = pd.concat(data_df_list, axis=0)
        # ###############################################
        return out_data_dict, mlm_para_label_list

    def token_insert_batch(self,
                           sp_token_tensor,
                           batch_data: dict,
                           mlm_label=None):
        """"""
        # pad_token = self.get_sp_token_batch('PAD')
        sos_token = torch.clone(sp_token_tensor[:, 1, :]).unsqueeze(1)
        eos_token = torch.clone(sp_token_tensor[:, 2, :]).unsqueeze(1)
        stp_token = torch.clone(sp_token_tensor[:, 3, :]).unsqueeze(1)
        cls_token = torch.clone(sp_token_tensor[:, 4, :]).unsqueeze(1)
        # mask_token = self.get_sp_token_batch('MASK')

        # #####################################################################################
        # sp_token_tensor = self.special_token_embedding.weight
        # sp_token_tensor = sp_token_tensor.detach().cpu().numpy()
        # sp_token_df = pd.DataFrame(data=sp_token_tensor,
        #                            index=['PAD', 'SOS', 'EOS', 'STP', 'CLS', 'MASK'])
        # #####################################################################################

        curr_segment_index = [0]

        # storage of segment index
        segment_index = []

        # storage of para tensor
        sequence_list = []
        paras_df_list = []

        token_idx_list = []
        for para_name, para_val in batch_data.items():
            temp_segment_list = []
            temp_para_df_list = []
            if curr_segment_index[0] == 0:
                temp_segment_list.append(torch.clone(cls_token))
                temp_segment_list.append(torch.clone(sos_token))

                token_idx_list.append('CLS')
                token_idx_list.append('SOS')

                # #################################################
                # temp_para_df_list.append(sp_token_df.loc[['CLS']])
                # temp_para_df_list.append(sp_token_df.loc[['SOS']])
                # ###################################################

            temp_segment_list.append(para_val)

            tmp_size = para_val.shape
            temp_idx_list = ['None'] * tmp_size[1]
            if mlm_label is not None:
                temp_para_mask_id_list = mlm_label[curr_segment_index[0]]
                for id, _, mask_type in temp_para_mask_id_list:
                    temp_idx_list[id] = mask_type
            token_idx_list = token_idx_list + temp_idx_list

            # #######################################################
            # temp_para_df = para_val[0, :, :].detach().cpu().numpy()
            # temp_para_df = pd.DataFrame(data=temp_para_df)
            # if mlm_label is not None:
            #     temp_para_mask_id_list = mlm_label[curr_segment_index[0]]
            #     for id, _, mask_type in temp_para_mask_id_list:
            #         temp_para_df = temp_para_df.rename(index={id: mask_type})
            # temp_para_df_list.append(temp_para_df)
            # #######################################################


            if curr_segment_index[0] != (self.max_num_seg - 1):
                temp_segment_list.append(torch.clone(stp_token))
                token_idx_list.append('STP')

                # #################################################
                # temp_para_df_list.append(sp_token_df.loc[['STP']])
                # ###################################################

            else:
                temp_segment_list.append(torch.clone(eos_token))
                token_idx_list.append('EOS')
                # #################################################
                # temp_para_df_list.append(sp_token_df.loc[['EOS']])
                # ###################################################

            # concat tokens
            temp_segment_arr = torch.cat(temp_segment_list, dim=1)
            # append segment arr to sequence list
            sequence_list.append(temp_segment_arr)


            # #######################################################
            # temp_para_with_sp_df = pd.concat(temp_para_df_list, axis=0)
            # paras_df_list.append(temp_para_with_sp_df)
            # #######################################################

            # generate segment index
            temp_segment_idx = curr_segment_index * temp_segment_arr.shape[1]
            segment_index = segment_index + temp_segment_idx
            curr_segment_index[0] = curr_segment_index[0] + 1

        temp_sequence_arr = torch.cat(sequence_list, dim=1)
        # ################################################
        # paras_df = pd.concat(paras_df_list, axis=0)
        # ################################################

        temp_mlm_pos_gt_list = None
        if mlm_label is not None:

            # get mask token ground true
            temp_mlm_gt_list = []
            for ls in mlm_label:
                for id, gt, _ in ls:
                    temp_mlm_gt_list.append(gt)

            # get mask token position
            temp_mlm_pos_list = []
            for i, index in enumerate(token_idx_list):
                if index in ['mask', 'rnd', 'org']:
                    temp_mlm_pos_list.append(i)

            temp_mlm_pos_gt_list = []
            for pos, gt in zip(temp_mlm_pos_list, temp_mlm_gt_list):
                temp_pos_gt_tuple = (pos, gt)
                temp_mlm_pos_gt_list.append(temp_pos_gt_tuple)

        return temp_sequence_arr, segment_index, temp_mlm_pos_gt_list

    def get_save_model(self):
        """"""
        attr_need = ['downstream_head',
                     'downstream_loss',
                     'batch_size',
                     'dropout',
                     'encoder_blk',
                     'mask_token_pre_head',
                     'mask_token_pred_loss',
                     'max_num_seg',
                     'max_num_token',
                     'model_batch_loss',
                     'model_name',
                     'n_head',
                     'n_hid',
                     'n_layer',
                     'next_para_pre_head',
                     'next_para_pre_loss',
                     'position_embedding',
                     'scale',
                     'segment_embedding',
                     'special_token_embedding',
                     'token_embedding',
                     'token_len']

        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    def get_out(self):
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss.item()

    def get_mtp_loss(self):
        """"""
        return self.MTP_batch_loss.item()

    def get_npp_loss(self):
        """"""
        return self.NPP_batch_loss.item()

    def get_batch_size(self, batch_data: dict):
        """"""
        para_list = list(batch_data.keys())
        first_para_data_size = batch_data[para_list[0]].shape
        self.batch_size = first_para_data_size[0]

    def get_sp_token_batch(self, sp_token: str):
        """"""
        # special token
        # 'PAD': 0,
        # 'SOS': 1,
        # 'EOS': 2,
        # 'STP': 3,
        # 'CLS': 4,
        # 'MASK': 5

        sp_token_idx = None
        if sp_token == 'PAD':
            sp_token_idx = 0
        elif sp_token == 'SOS':
            sp_token_idx = 1
        elif sp_token == 'EOS':
            sp_token_idx = 2
        elif sp_token == 'STP':
            sp_token_idx = 3
        elif sp_token == 'CLS':
            sp_token_idx = 4
        elif sp_token == 'MASK':
            sp_token_idx = 5
        else:
            raise ValueError('Undefined Token')

        num_sp_token = self.special_token_embedding.weight.shape[0]
        if (sp_token_idx <= num_sp_token) and (sp_token_idx >= 0):
            temp_sp_idx_tensor = torch.tensor([sp_token_idx],
                                                 dtype=torch.long,
                                                 device=self.device)
            temp_sp_token = self.special_token_embedding(temp_sp_idx_tensor)
            return temp_sp_token
        else:
            raise ValueError('Special Token Index Error')


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    ###################################################################################################################
    # std import
    # import os
    # import sys
    ###################################################################################################################
    # third party import
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    ###################################################################################################################
    # app specific import
    # import utility_function
    import preprocess
    import init_train_module

    ###################################################################################################################
    # set the random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    ###################################################################################################################
    # set device
    m_device = init_train_module.init_device('gpu', 0)
    ###################################################################################################################
    # set the data set parameters
    m_data_set_path = '.\\pik\\22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
    m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'

    m_rnd_token = init_train_module.rnd_token_loader(m_rnd_token_path)
    m_rnd_para = init_train_module.rnd_para_loader(m_rnd_para_path)

    m_train_mode = 'pretrain'  # ('pretrain', 'train', 'test', 'finetune')
    #           len(batch_size)
    # pre-train        1
    # other            3
    batch_size = [256]
    m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, batch_size, False)
    ###################################################################################################################
    # set preprocessing
    m_prepro_param = {
        'num_classes': 8,
        'token_tuple': (32, False, 1),
        'rnd_para_dict': m_rnd_para
    }
    m_prepro = preprocess.MyMultiBertModelProcessing(**m_prepro_param)

    # preprocess parameter for baseline
    # m_preprocess_param = {
    #     'num_classes': 8,
    # }
    # m_preprocessor = preprocess.BaseProcessing(**m_preprocess_param)
    ###################################################################################################################
    # model parameter for MultiBert
    m_model_param = {
        'device': m_device,
        'token_len': 32,
        'rnd_token': m_rnd_token,
        'max_num_seg': 5,
        'max_num_token': 100,
        'embedding_dim': 16,
        'n_layer': 3,
        'n_head': 4,
        'n_hid': 256
    }
    m_init_model = MyMultiBertModel(**m_model_param).to(m_device)

    # model parameter for baseline
    # m_model = model_define.BaseLine_MLP
    # m_model_param = {
    #     'in_dim': 1021,
    #     'num_cls': 8,
    #     'loss_func': nn.CrossEntropyLoss(),
    # }
    ###################################################################################################################
    # optimizer and scheduler set up
    m_optimizer_param = {
        'optimizer_name': 'Adam',
        'lr': 0.0001,
        'betas': (0.9, 0.98),
        'eps': 1e-9,
        'weight_decay': 0,
        'amsgrad': False,
    }
    m_scheduler_param = {
        'scheduler name': 'StepLR',
        'step_size': 16,
        'gamma': 0.95,
        'last_epoch': -1,
        'verbose': False
    }
    m_opt, m_sch = init_train_module.init_optimaizer_scheduler(m_init_model, m_optimizer_param, m_scheduler_param)
    ###################################################################################################################
    # train
    m_log_dir = '.\\log'
    ###################################################################################################################
    # collect hyper parameter
    m_hyper_param = {
        'train_mode': m_train_mode,
        'data_set': m_data_set_path,
        'rnd_token': m_rnd_token_path,
        'rnd_para': m_rnd_para_path,
        'batch_size': batch_size[0],
        'token_len': m_prepro_param['token_tuple'][0],
        'model_name': m_init_model.model_name,
        'max_num_seg': m_model_param['max_num_seg'],
        'embedding_dim': m_model_param['embedding_dim'],
        'n_layer': m_model_param['n_layer'],
        'n_head': m_model_param['n_head'],
        'n_hid': m_model_param['n_hid'],
    }

    add_graph_flag = False
    with SummaryWriter(log_dir=m_log_dir) as writer:
        for epoch_idx in range(32):
            # set model to train mode
            m_init_model.train()
            for batch_idx, data_label in enumerate(m_data_loader_dict[m_train_mode]):
                with torch.autograd.set_detect_anomaly(True):
                    # pre-process
                    input_list = m_prepro.pro(data_label,
                                              m_train_mode,
                                              m_device)

                    # if not add_graph_flag:
                    #     writer.add_graph(m_init_model,
                    #                      [])
                    #     add_graph_flag = True

                    model_loss = m_init_model(*input_list)

                    # g = make_dot(model_loss,
                    #              params=dict(m_init_model.named_parameters()),
                    #              show_attrs=True,
                    #              show_saved=True)
                    # g.render(filename='graph', view=False)

                    m_opt.zero_grad()
                    model_loss.backward(retain_graph=True)
                    m_opt.step()

                print(f'epoch: {epoch_idx},batch: {batch_idx}')
