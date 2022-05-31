#!/usr/bin/env python
# -*- coding: utf-8 -*-
import optuna
import torch.nn as nn

import preprocess
import multi_bert_model
import init_train_module

def fine_tune_func(trial, trial_root_path, model_dir, experiment_start_time):
    ###################################################################################################################
    # set device
    m_device = init_train_module.init_device('gpu', 0)
    ###################################################################################################################
    # set the data set parameters
    m_data_set_path = '.\\pik\\test_2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_train_mode = 'finetune'  # ('pretrain', 'train', 'test', 'finetune')
    # trial number
    current_trial_id = f'{m_train_mode}_trial_{trial.number}'
    ###################################################################################################################
    m_model_step = 1
    trials_pretrain = os.listdir(model_dir)
    for trial_pretrain_id, trial_pretrain_name in enumerate(trials_pretrain):
        temp_trial_pretrain_path = os.path.join(model_dir, trial_pretrain_name)
        temp_pretrain_model_list = os.listdir(temp_trial_pretrain_path)
        temp_num_trial_model = len(temp_pretrain_model_list)

        if temp_num_trial_model == 0:
            raise FileExistsError('No Pretrain Model Exist.')
        elif temp_num_trial_model > 1:
            raise FileExistsError('To Much Pretrain Model Exist.')
        else:
            pass

        current_pretrain_model_name = temp_pretrain_model_list[0]
        ################################################################################################################
        # parameter analysis
        param_list = trial_pretrain_name.split('_')
        m_bsz = int(param_list[0].split('-')[-1])
        m_tlen = int(param_list[1].split('-')[-1])
        m_nlayer = int(param_list[2].split('-')[-1])
        m_nhead = int(param_list[3].split('-')[-1])
        m_nhid = int(param_list[4].split('-')[-1])
        ################################################################################################################
        pretrain_experiment_time = model_dir.split('\\')[-2]

        # train
        m_log_dir = os.path.join(trial_root_path,
                                 m_train_mode,
                                 experiment_start_time,
                                 current_trial_id,
                                 f'{pretrain_experiment_time}_'
                                 f'{trial_pretrain_name}_'
                                 f'{current_pretrain_model_name}')
        m_model_dir = os.path.join(temp_trial_pretrain_path, current_pretrain_model_name, 'models')
    ###################################################################################################################
        in_token_len = m_tlen
        m_rnd_token, m_rnd_para = init_train_module.get_rnd_token_para(m_data_set_path,
                                                                       in_token_len,
                                                                       m_device)


        #           len(batch_size)
        # pre-train        1
        # other            3
        batch_size = [m_bsz, 100, 100]
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
            'max_num_seg': 1,
            'max_num_token': 1,
            'n_layer': m_nlayer,
            'n_head': m_nhead,
            'n_hid': m_nhid,
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
        # set loss function
        down_Loss_fn = nn.CrossEntropyLoss()
        m_num_class = 8

        m_loss_fn_list = [down_Loss_fn,]

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
                                                model_dir=m_model_dir,
                                                optimizer_param=m_optimizer_param,
                                                scheduler_param=m_scheduler_param,
                                                num_class=m_num_class,
                                                pretrain_step=m_model_step)

        # train_mode:('pretrain', 'train')
        metric = 0.0
        m_trainer.run()
        # return metric


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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # RANDOM_SEED = 42
    # np.random.seed(RANDOM_SEED)
    # torch.manual_seed(RANDOM_SEED)
    #
    ####################################################################################################################
    writer_dir = '.\\log'
    # set model path for finetune
    model_dir = '.\\log\\pretrain\\20220517-084346\\'
    ###################################################################################################################

    n_trials = 1
    m_sampler = optuna.samplers.RandomSampler()
    m_pruner = optuna.pruners.NopPruner()
    m_direction = optuna.study.StudyDirection.MINIMIZE

    study = optuna.create_study(sampler=m_sampler, pruner=m_pruner, direction=m_direction)
    study.optimize(lambda trial: fine_tune_func(trial, writer_dir, model_dir, data_time_str),
                   n_trials=n_trials, timeout=None, gc_after_trial=True)
    # get trials result
    exp_res = study.trials_dataframe()
    exp_res.to_csv(os.path.join(writer_dir, f'{data_time_str}_Trials_DataFrame.csv'))

    # save study
    with open(os.path.join(writer_dir, f'{data_time_str}_Study.pkl'), 'wb') as f:
        pickle.dump(study, f)
    sys.exit(0)