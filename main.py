#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':

    # std import
    import os
    import sys
    from datetime import datetime

    # third party import
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    # app specific import
    import utility_function
    import dataset_and_dataloader
    import opt
    import model_define
    # import run

    # set the random seed
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.environ['OMP_NUM_THREADS'] = '1'

    data_file_base = '.\\pik'
    curr_file_name = 'test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_data_file_path = os.path.join(data_file_base, curr_file_name)
    m_data_dict = utility_function.read_pickle_file(m_data_file_path)

    # set the data set parameters
    len_to_token = 16
    batch_size = 32
    is_shuffle = True
    workers = 3
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

    # end run
    sys.exit(0)
