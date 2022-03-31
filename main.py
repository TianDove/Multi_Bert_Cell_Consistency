#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    ###################################################################################################################
    # std import
    import os
    import sys
    from datetime import datetime
    ###################################################################################################################
    # third party import
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    ###################################################################################################################
    # app specific import
    import utility_function
    import dataset_and_dataloader
    import opt
    import model_define
    import model_run
    ###################################################################################################################
    # set the random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    ###################################################################################################################
    # set device
    USE_GPU = True
    if USE_GPU:
        device = utility_function.try_gpu()
    else:
        device = torch.device('cpu')
    ###################################################################################################################
    # load dataset file
    data_file_base_path = '.\\pik'
    load_data_dict = {}
    data_file_name = {
        'data': 'test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle',
        'rnd_token': '2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle',
        'rnd_para': '2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'
    }

    for data_key, data_path in iter(data_file_name.items()):
        temp_load_path = os.path.join(data_file_base_path, data_path)
        load_data_dict[data_key] = utility_function.read_pickle_file(temp_load_path)

    # to tensor
    load_data_dict['rnd_token'] = torch.from_numpy(load_data_dict['rnd_token']).to(dtype=torch.float32)
    for para_name, para_table in iter(load_data_dict['rnd_para'].items()):
        load_data_dict['rnd_para'][para_name] = torch.from_numpy(para_table).to(dtype=torch.float32)
    ###################################################################################################################
    # set the data set parameters
    len_to_token = 32
    token_tuple = (len_to_token, False, 1)
    m_tokenizer = utility_function.Tokenizer(token_tuple)
    num_epochs = 512
    batch_size = 8
    is_shuffle = True
    workers = 1
    pin_memory = True

    # ('pretrain', 'train', 'val', 'test')
    data_set_dict = {}
    data_set_type = ('pretrain', 'train', 'val', 'test')
    data_dict = load_data_dict['data']
    for t in iter(data_set_type):
        data_set_dict[t] = dataset_and_dataloader.CellDataLoader.creat_data_loader(data_dict=data_dict,
                                                                                   type_data_set=t,
                                                                                   batch_sz=batch_size,
                                                                                   is_shuffle=is_shuffle,
                                                                                   num_of_worker=workers,
                                                                                   pin_memory=pin_memory)
    ###################################################################################################################
    # model init
    model_para_dict = {
        'token_tuple-in_len': len_to_token,
        'token_tuple-overlap': False,
        'token_tuple-step': 1,
        'batch_size': batch_size,
        'embedding_token_dim': 16,
        'max_num_seg': 5,
        'max_token': 10000,
        'encoder_para-layers': 3,
        'encoder_para-nhead': 4,
        'encoder_para-hid': 256,
    }
    m_model = model_define.MyMulitBERT(token_tuple=(model_para_dict['token_tuple-in_len'],
                                                    model_para_dict['token_tuple-overlap'],
                                                    model_para_dict['token_tuple-step']),
                                       rnd_token_table=load_data_dict['rnd_token'],
                                       batch_size=model_para_dict['batch_size'],
                                       embedding_token_dim=model_para_dict['embedding_token_dim'],
                                       max_num_seg=model_para_dict['max_num_seg'],
                                       max_token=model_para_dict['max_token'],
                                       encoder_para=(model_para_dict['encoder_para-layers'],
                                                     model_para_dict['encoder_para-nhead'],
                                                     model_para_dict['encoder_para-hid']),
                                       device=device)
    m_model = utility_function.init_model(m_model).to(device)
    nsp_loss = torch.nn.CrossEntropyLoss()
    mlm_loss = torch.nn.MSELoss()
    pretrain_loss_func_dict = {
        'nsp_loss': nsp_loss,
        'mlm_loss': mlm_loss
    }
    ###################################################################################################################
    # optimizer and scheduler set up
    optimizer_para = {
        'optimizer_name': 'Adam',
        'lr': 0.0001,
        'betas': (0.9, 0.98),
        'eps': 1e-9,
        'weight_decay': 0,
        'amsgrad': False,
    }
    scheduler_para = {
        'scheduler name': 'StepLR',
        'step_size': 10,
        'gamma': 0.95,
        'last_epoch': -1,
        'verbose': False
    }
    op_sch_dict = opt.optimizer_select_and_init(m_model, optimizer_para, scheduler_para)
    optimizer_para.pop('betas')  # del for hyper-parameter logging
    ###################################################################################################################
    # gether hyper parameters
    hyper_para_dict = {
        'data_time': datetime.now().strftime("%Y%m%d-%H%M%S"),
        'log_dir': './/log//',
        'train_mode': 'pretrain',
        'model_name': m_model.model_name,
        'device_idx': device.index,
        'device': device.type,
        'data_set': data_file_name['data'],
        'len_to_token': len_to_token,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        **model_para_dict,
        **optimizer_para,
        **scheduler_para
    }
    ###################################################################################################################
    # train

    writer_dir = os.path.join(hyper_para_dict['log_dir'], f'{m_model.model_name}_{hyper_para_dict["data_time"]}')
    with SummaryWriter(log_dir=writer_dir) as writer:
        m_run = model_run.ClsRun(train_mode=hyper_para_dict['train_mode'],
                                 data_time=hyper_para_dict['data_time'],
                                 data_set_dic=data_set_dict,
                                 rnd_para_dict=load_data_dict['rnd_para'],
                                 tokenizer=m_tokenizer,
                                 n_epochs=num_epochs,
                                 model=m_model,
                                 loss_fn=pretrain_loss_func_dict,
                                 writer=writer,
                                 optimizer=op_sch_dict['optimizer'],
                                 scheduler=op_sch_dict['scheduler'],
                                 device=device,
                                 hyper_para_dict=hyper_para_dict)
        m_run.run()
    sys.exit(0)
