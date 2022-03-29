#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# import run
###################################################################################################################
# set the random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
###################################################################################################################
# set device
USE_GPU = False
if USE_GPU:
    device = utility_function.try_gpu()
else:
    device = torch.device('cpu')
###################################################################################################################
# load dataset file
m_data_file_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
m_data_dict = utility_function.read_pickle_file(m_data_file_path)

# load random token file
m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
m_rnd_token = utility_function.read_pickle_file(m_rnd_token_path)

# load random para file
m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'
m_rnd_para = utility_function.read_pickle_file(m_rnd_para_path)
###################################################################################################################
# set the data set parameters
len_to_token = 32
batch_size = 64
is_shuffle = True
workers = 1
pin_memory = True

# ('pretrain', 'train', 'val', 'test')
pretrain_loader = dataset_and_dataloader.CellDataLoader.creat_data_loader(m_data_dict, 'pretrain',
                                                                          batch_size, is_shuffle,
                                                                          workers, pin_memory)
train_loader = dataset_and_dataloader.CellDataLoader.creat_data_loader(m_data_dict, 'train',
                                                                       batch_size, is_shuffle,
                                                                       workers, pin_memory)
val_loader = dataset_and_dataloader.CellDataLoader.creat_data_loader(m_data_dict, 'val',
                                                                     batch_size, is_shuffle,
                                                                     workers, pin_memory)
test_loader = dataset_and_dataloader.CellDataLoader.creat_data_loader(m_data_dict, 'test',
                                                                      batch_size, is_shuffle,
                                                                      workers, pin_memory)

data_set_dict = {
    'pretrain': pretrain_loader,
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}
###################################################################################################################
# model init
m_model = model_define.MyMulitBERT(token_tuple=(len_to_token, False, 1),
                                   rnd_token_table=m_rnd_token,
                                   rnd_para_table=m_rnd_para,
                                   batch_size=batch_size,
                                   embedding_token_dim=16,
                                   max_num_seg=5,
                                   max_token=10000,
                                   encoder_para=(3, 4, 256))
m_model.to(device)
nsp_loss = torch.nn.MSELoss()
mlm_loss = torch.nn.CrossEntropyLoss()
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
###################################################################################################################
data_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = './/log//'
writer_dir = os.path.join(log_dir, f'{m_model.model_name}_{data_time_str}')
with SummaryWriter(log_dir=writer_dir) as writer:
    pass
sys.exit(0)
