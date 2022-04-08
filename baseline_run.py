#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
    ###################################################################################################################
    # set the random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    ###################################################################################################################
    # set device
    m_device_type = 'gpu'
    ###################################################################################################################
    # dataset file path
    m_data_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed_cat_pretrain.pickle'
    ###################################################################################################################
    # set the data set parameters
    m_batch_size = 3
    m_workers = 1
    ###################################################################################################################
    # set preprocessing
    # preprocess parameter for baseline
    m_preprocess_param = {
        'num_classes': 8,
    }
    m_preprocessor = preprocess.BaseProcessing(**m_preprocess_param)
    ###################################################################################################################
    # model parameter for baseline
    m_model = model_define.BaseLine_MLP
    m_model_param = {
        'in_dim': 1021,
        'num_cls': 8,
        'loss_func': nn.CrossEntropyLoss(),
    }
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
        'step_size': 10,
        'gamma': 0.95,
        'last_epoch': -1,
        'verbose': False
    }
    ###################################################################################################################
    # train
    m_log_dir = '.\\log'
    ###################################################################################################################
    m_trainer = trainer.Trainer(device_type=m_device_type,
                                log_dir=m_log_dir,
                                data_path=m_data_path,
                                batch_size=m_batch_size,
                                workers=m_workers,
                                model=m_model,
                                model_param=m_model_param,
                                optimizer_param=m_optimizer_param,
                                scheduler_param=m_scheduler_param)

    # train_mode:('pretrain', 'train')
    m_trainer.run(train_mode='train',
                  num_epoch=128,
                  preprocessor=m_preprocessor)
    sys.exit(0)
