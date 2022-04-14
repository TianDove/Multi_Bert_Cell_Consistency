#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    ###################################################################################################################
    # std import
    import os
    import sys
    ###################################################################################################################
    # third party import
    import numpy as np
    import torch
    import torch.nn as nn
    ###################################################################################################################
    # app specific import
    import utility_function
    import preprocess
    import model_define
    import trainer
    import init_train_module
    import mymodel

    ###################################################################################################################
    # set the random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    ###################################################################################################################
    m_device = init_train_module.init_device('gpu', 0)

    m_data_set_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
    m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'

    m_rnd_token = init_train_module.rnd_token_loader(m_rnd_token_path, m_device)
    m_rnd_para = init_train_module.rnd_para_loader(m_rnd_para_path)

    m_train_mode = 'train'  # ('pretrain', 'train', 'test', 'finetune')
    m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, [8, 8, 8])

    m_prepro_param = {
        'num_classes': 8,
        'token_tuple': (32, False, 1),
        'rnd_para_dict': m_rnd_para
    }
    m_prepro = preprocess.MyMultiBertModelProcessing(**m_prepro_param)

    m_model = mymodel.MyMultiBertModel
    m_model_param = {
        'token_len': 32,
        'rnd_token':m_rnd_token,
        'max_num_seg': 5,
        'max_num_token': 100,
        'embedding_dim': 16,
        'n_layer':3,
        'n_head': 4,
        'n_hid': 256
    }
    m_init_model = init_train_module.init_model(m_model, m_model_param, m_device)

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

    for b_idx, data_label in enumerate(m_data_loader_dict[m_train_mode]):
        temp_data_label = data_label
        data_list = m_prepro.pro(temp_data_label, m_train_mode, m_device)
        tmp = m_init_model(*data_list, train_mode=m_train_mode)


