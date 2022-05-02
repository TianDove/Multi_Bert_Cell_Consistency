#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    # ###################################################################################################################
    # # std import
    import os
    import sys
    import datetime
    import pickle
    # ###################################################################################################################
    # # third party import
    import numpy as np
    import torch
    import torch.nn as nn
    # ###################################################################################################################
    # # app specific import
    # import utility_function
    import preprocess
    import init_train_module
    import model_define
    #
    # set device
    m_device = init_train_module.init_device('gpu', 0)
    ###################################################################################################################
    # set the data set parameters
    m_data_set_path = '.\\pik\\22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'

    m_train_mode = 'test'  # ('pretrain', 'train', 'test', 'finetune')

    #           len(batch_size)
    # pre-train        1
    # other            3
    batch_size = [64, 100, 100]
    m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, batch_size)
    ###################################################################################################################
    # set preprocessing
    # preprocess parameter for baseline
    m_prepro_param = {
        'num_classes': 8,
    }
    m_prepro = preprocess.BaseProcessing(**m_prepro_param)
    ###################################################################################################################
    # model parameter for baseline
    m_model = model_define.BaseLine_ResNet
    m_model_param = {
        'in_dim': 1,
        'num_cls': m_prepro_param['num_classes'],
    }
    m_init_model = init_train_module.init_model(m_model, m_model_param, m_device)
    ###################################################################################################################
    # train
    m_log_dir = os.path.join('.\\log', m_train_mode)
    # set model path for finetune
    m_model_dir = '.\\log\\train\\train_BaseLine_ResNet_2022-04-28-12-57-43\\models'
    ###################################################################################################################
    m_trainer = init_train_module.Model_Run(device=m_device,
                                            train_mode=m_train_mode,
                                            data_loader=m_data_loader_dict,
                                            preprocessor=m_prepro,
                                            model=m_init_model,
                                            log_dir=m_log_dir,
                                            model_dir=m_model_dir,
                                            num_class=m_prepro_param['num_classes'])

    # train_mode:('pretrain', 'train')
    m_trainer.run()
