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
    import multi_bert_model
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
    m_data_set_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
    m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'

    m_rnd_token = init_train_module.rnd_token_loader(m_rnd_token_path)
    m_rnd_para = init_train_module.rnd_para_loader(m_rnd_para_path)

    m_train_mode = 'test'  # ('pretrain', 'train', 'test', 'finetune')
    #           len(batch_size)
    # pre-train        1
    # other            3
    batch_size = [32, 32, 32]
    m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, batch_size)
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
    m_model = multi_bert_model.MyMultiBertModel
    m_model_param = {
        'device': m_device,
        'token_len': m_prepro_param['token_tuple'][0],
        'rnd_token': m_rnd_token,
        'max_num_seg': 1,
        'max_num_token': 1,
        'embedding_dim': 4,
        'n_layer': 1,
        'n_head': 2,
        'n_hid': 16
    }
    m_init_model = init_train_module.init_model(m_model, m_model_param, m_device)

    # model parameter for baseline
    # m_model = model_define.BaseLine_MLP
    # m_model_param = {
    #     'in_dim': 1021,
    #     'num_cls': 8,
    #     'loss_func': nn.CrossEntropyLoss(),
    # }
    ###################################################################################################################
    # train
    m_log_dir = '.\\log'
    ###################################################################################################################
    # set loss function
    m_num_class = 8
    ###################################################################################################################
    # set model path for finetune
    m_model_dir = '.\\log\\MyMultiBertModel_2022-04-26-09-35-36\\models'
    ###################################################################################################################
    m_trainer = init_train_module.Model_Run(device=m_device,
                                            train_mode=m_train_mode,
                                            data_loader=m_data_loader_dict,
                                            preprocessor=m_prepro,
                                            model=m_init_model,
                                            log_dir=m_log_dir,
                                            model_dir=m_model_dir,
                                            num_class=m_num_class)

    # train_mode:('pretrain', 'train')
    m_trainer.run()
    sys.exit(0)
