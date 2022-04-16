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
    m_data_set_path = '.\\pik\\22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
    m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'

    m_rnd_token = init_train_module.rnd_token_loader(m_rnd_token_path)
    m_rnd_para = init_train_module.rnd_para_loader(m_rnd_para_path)

    m_train_mode = 'pretrain'  # ('pretrain', 'train', 'test', 'finetune')
    #           len(batch_size)
    # pre-train        1
    # other            3
    batch_size = [4]
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
        'token_len': 32,
        'rnd_token': m_rnd_token,
        'max_num_seg': 5,
        'max_num_token': 100,
        'embedding_dim': 16,
        'n_layer': 3,
        'n_head': 4,
        'n_hid': 256
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
    ###################################################################################################################
    m_trainer = init_train_module.Model_Run(device=m_device,
                                            train_mode=m_train_mode,
                                            num_epoch=1024,
                                            data_loader=m_data_loader_dict,
                                            preprocessor=m_prepro,
                                            model=m_init_model,
                                            optimizer=m_opt,
                                            scheduler=m_sch,
                                            log_dir=m_log_dir,
                                            hyper_param=m_hyper_param)

    # train_mode:('pretrain', 'train')
    m_trainer.run()
    sys.exit(0)
