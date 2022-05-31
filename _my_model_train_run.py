#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import optuna
import torch.nn as nn

import preprocess
import multi_bert_model
import init_train_module

def train_func(trial, trial_root_path, experiment_start_time, *args, **kwargs):
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

    m_train_mode = 'train'  # ('pretrain', 'train', 'test', 'finetune')

    # trial number
    current_trial_id = f'{m_train_mode}_trial_{trial.number}'
    #           len(batch_size)
    # pre-train        1
    # other            3
    batch_size = [64, 100, 100]
    m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, batch_size)
    ###################################################################################################################
    # set preprocessing
    # para_name ('ch1v', 'ch2v', 'dcv', 'ch3v', 'ch3c') all
    # num_token (  6,      6,      8,      7,      7, ) 41
    params_type = trial.suggest_categorical('params and num_token', (0, 1, 2, 3, 4, 5))

    if params_type == 0:
        m_params = ('ch1v',)
        m_num_token = 6
    elif params_type == 1:
        m_params = ('ch2v',)
        m_num_token = 6
    elif params_type == 2:
        m_params = ('dcv',)
        m_num_token = 8
    elif params_type == 3:
        m_params = ('ch3v',)
        m_num_token = 7
    elif params_type == 4:
        m_params = ('ch3c',)
        m_num_token = 7
    elif params_type == 5:
        m_params = ('ch1v', 'ch2v', 'dcv', 'ch3v', 'ch3c')
        m_num_token = 41
    else:
        raise ValueError('Params Type Error.')

    m_prepro_param = {
        'num_classes': 8,
        'token_tuple': (32, False, 1),
        'rnd_para_dict': m_rnd_para,
        'params': m_params,
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
        'num_token': m_num_token,
        # 'n_layer': 3,
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
    train_save_name = None
    if len(m_params) == 1:
        train_save_name = m_params[-1]
    elif len(m_params) == 5:
        train_save_name = 'all'
    else:
        pass
    m_log_dir = os.path.join(trial_root_path, f'{m_train_mode}_{train_save_name}', experiment_start_time, current_trial_id)
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
    # set loss function
    Down_Loss_fn = nn.CrossEntropyLoss()

    m_loss_fn_list = [Down_Loss_fn, ]
    ###################################################################################################################
    m_trainer = init_train_module.Model_Run(device=m_device,
                                            train_mode=m_train_mode,
                                            num_epoch=2,
                                            data_loader=m_data_loader_dict,
                                            preprocessor=m_prepro,
                                            model=m_init_model,
                                            loss_fn_list=m_loss_fn_list,
                                            optimizer=m_opt,
                                            scheduler=m_sch,
                                            log_dir=m_log_dir,
                                            hyper_param=m_hyper_param,
                                            num_class=m_prepro_param['num_classes'])

    # train_mode:('pretrain', 'train')
    metric = 0.0
    m_trainer.run()
    return metric

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
    # import torch.nn as nn
    # ###################################################################################################################
    # # app specific import
    # import utility_function
    # import preprocess
    # import model_define
    # import multi_bert_model
    # import init_train_module
    #
    # ###################################################################################################################
    # set the random seed
    data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # RANDOM_SEED = 42
    # np.random.seed(RANDOM_SEED)
    # torch.manual_seed(RANDOM_SEED)
    #
    ####################################################################################################################
    writer_dir = '.\\log'

    # para_name ('ch1v', 'ch2v', 'dcv', 'ch3v', 'ch3c') all
    # num_token (  6,      6,      8,      7,      7, ) 41
    m_params_num_token = {
        'params_type':(0, 1, 2, 3, 4, 5)
    }

    n_trials = len(m_params_num_token)
    # m_sampler = optuna.samplers.GridSampler()
    m_pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(sampler=None, pruner=m_pruner, direction="maximize")
    study.optimize(lambda trial: train_func(trial, writer_dir, data_time_str),
                   n_trials=n_trials, show_progress_bar=True)
    # get trials result
    exp_res = study.trials_dataframe()
    exp_res.to_csv(os.path.join(writer_dir, f'{data_time_str}_Trials_DataFrame.csv'))

    # save study
    with open(os.path.join(writer_dir, f'{data_time_str}_Study.pkl'), 'wb') as f:
        pickle.dump(study, f)
    sys.exit(0)
