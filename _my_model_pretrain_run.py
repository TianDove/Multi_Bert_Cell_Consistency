#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import optuna
import torch.nn as nn

import preprocess
import multi_bert_model
import init_train_module


def pretrain_func(trial, trial_root_path, experiment_start_time, train_mode):
    ###################################################################################################################

    # set device
    m_device = init_train_module.init_device('gpu', 0)
    ###################################################################################################################
    # set the data set parameters
    m_data_set_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'

    in_token_len = trial.suggest_int('tokenlen', 32, 128)
    m_rnd_token, m_rnd_para = init_train_module.get_rnd_token_para(m_data_set_path,
                                                                   in_token_len,
                                                                   m_device)

    m_train_mode = train_mode  # ('pretrain', 'train', 'test', 'finetune')

    # trial number
    current_trial_id = f'{m_train_mode}_trial_{trial.number}'
    #           len(batch_size)
    # pre-train        1
    # other            3
    m_epoch = 512
    batch_size = [trial.suggest_int('bsz', 2, 2048), ]
    m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, batch_size)
    ###################################################################################################################
    # set preprocessing
    m_prepro_param = {
        'num_classes': 8,
        'token_tuple': (in_token_len, False, 1),
        'rnd_para_dict': m_rnd_para
    }
    m_prepro = preprocess.MyMultiBertModelProcessing(**m_prepro_param)

    num_in_token = m_prepro.cal_num_in_token()

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
        'max_num_seg': 5,
        'max_num_token': 1000,
        # 'n_layer': 3,
        # 'n_head': 4,
        # 'n_hid': 256
        'n_layer': trial.suggest_int('nlayer', 1, 24),
        'n_head': trial.suggest_int('nhead', 1, 32),
        'n_hid': trial.suggest_int('nhid', 2, 2048),
        'num_token': num_in_token,
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
    m_bsz = batch_size[0]
    m_tlen = m_prepro_param['token_tuple'][0]
    m_nlayer = m_model_param['n_layer']
    m_nhead = m_model_param['n_head']
    m_nhid = m_model_param['n_hid']

    m_tune_name = f'bsz-{m_bsz}_tlen-{m_tlen}_nlayer-{m_nlayer}_nhead-{m_nhead}_nhid-{m_nhid}'
    m_log_dir = os.path.join(trial_root_path, m_tune_name)
    ###################################################################################################################
    # collect hyper parameter
    m_hyper_param = {
        'train_mode': m_train_mode,
        'data_set': m_data_set_path,
        'batch_size': batch_size[0],
        'token_len': m_prepro_param['token_tuple'][0],
        'model_name': m_init_model.model_name,
        'max_num_seg': m_model_param['max_num_seg'],
        'n_layer': m_model_param['n_layer'],
        'n_head': m_model_param['n_head'],
        'n_hid': m_model_param['n_hid'],
    }
    ###################################################################################################################
    # set loss function
    NPP_Loss_fn = nn.CrossEntropyLoss()
    MTP_Loss_fn = nn.MSELoss()

    m_loss_fn_list = [NPP_Loss_fn, MTP_Loss_fn]
    ###################################################################################################################
    m_trainer = init_train_module.Model_Run(device=m_device,
                                            train_mode=m_train_mode,
                                            num_epoch=m_epoch,
                                            data_loader=m_data_loader_dict,
                                            preprocessor=m_prepro,
                                            model=m_init_model,
                                            loss_fn_list=m_loss_fn_list,
                                            optimizer=m_opt,
                                            scheduler=m_sch,
                                            log_dir=m_log_dir,
                                            hyper_param=m_hyper_param)

    # train_mode:('pretrain', 'train')
    metric = m_trainer.run()
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
    # import numpy as np
    # import torch
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
    m_train_mode = 'pretrain'  # ('pretrain', 'train', 'test', 'finetune')
    # RANDOM_SEED = 42
    # np.random.seed(RANDOM_SEED)
    # torch.manual_seed(RANDOM_SEED)
    #
    ####################################################################################################################
    writer_dir = f'.\\log\\{m_train_mode}\\{data_time_str}'

    n_trials = 8

    m_search_space = {
        # 'bsz': [2, 8, 32, 256, 512, 1024],
        'bsz': [64, ],  # 2 - 2048
        'tokenlen': [96, ],  # 64, 72, 96, 128],  # 32 - 128
        'nlayer': [3, ],  # 1 - 24
        'nhead': [4, ],  # 1 - 32
        'nhid': [128, ],  # 2 - 2048
    }
    m_sampler = optuna.samplers.GridSampler(search_space=m_search_space)
    m_pruner = optuna.pruners.NopPruner()
    m_direction = optuna.study.StudyDirection.MINIMIZE

    study = optuna.create_study(sampler=m_sampler, pruner=m_pruner, direction=m_direction)
    study.optimize(lambda trial: pretrain_func(trial, writer_dir, data_time_str, m_train_mode),
                   n_trials=n_trials, timeout=None, gc_after_trial=True)
    # get trials result
    exp_res = study.trials_dataframe()
    exp_res.to_csv(os.path.join(writer_dir, f'{data_time_str}_Trials_DataFrame.csv'))

    # save study
    with open(os.path.join(writer_dir, f'{data_time_str}_Study.pkl'), 'wb') as f:
        pickle.dump(study, f)
    sys.exit(0)
