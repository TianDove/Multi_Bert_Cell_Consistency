#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import optim_sche
import utility_function
import dataset_and_dataloader


def init_device(device_name: str, gpu_idx=None):
    """"""
    tmp_device = None
    if device_name == 'cpu':
        tmp_device = torch.device('cpu')
    elif device_name == 'gpu':
        if gpu_idx is not None:
            tmp_device = utility_function.try_gpu(gpu_idx)
        else:
            tmp_device = utility_function.try_all_gpus()

    return tmp_device


def rnd_token_loader(rnd_token_file_path: str, device: torch.device = None) -> torch.tensor:
    """"""
    if os.path.exists(rnd_token_file_path):
        temp_rnd_token_array = utility_function.read_pickle_file(rnd_token_file_path)
        temp_rnd_token_tensor = torch.from_numpy(temp_rnd_token_array).to(dtype=torch.float32)
        if device is not None:
            temp_rnd_token_tensor = temp_rnd_token_tensor.to(device)
        return temp_rnd_token_tensor
    else:
        raise FileNotFoundError


def rnd_para_loader(rnd_para_file_path: str, device: torch.device = None) -> dict:
    """"""
    if os.path.exists(rnd_para_file_path):
        temp_rnd_para_tensor_dict = {}
        temp_rnd_para_dict = utility_function.read_pickle_file(rnd_para_file_path)
        for para_name, para_val in temp_rnd_para_dict.items():
            temp_para_val = torch.from_numpy(para_val).to(dtype=torch.float32)
            if device is not None:
                temp_para_val = temp_para_val.to(device)

            temp_rnd_para_tensor_dict[para_name] = temp_para_val
        return temp_rnd_para_tensor_dict
    else:
        raise FileNotFoundError


def init_data_loader_dict(data_set_file_path: str,
                          train_mode: str,
                          batch_sz: list,
                          is_shuffle: bool = True,
                          num_of_worker: int = 1) -> dict:
    """"""
    if os.path.exists(data_set_file_path):
        temp_data_set_dict = utility_function.read_pickle_file(data_set_file_path)

        # decide whether use pretrian data set
        set_name_list = list(temp_data_set_dict.keys())
        if train_mode == 'pretrain':
            temp_ch_data_set_dict = {train_mode: temp_data_set_dict[train_mode]}
        elif train_mode in ['train', 'test', 'finetune']:
            temp_train_set = temp_data_set_dict['train']
            temp_pretrain_set = temp_data_set_dict['pretrain']
            temp_pretrain_cat_train_dict = {**temp_pretrain_set, **temp_train_set}
            del temp_data_set_dict['pretrain']
            temp_data_set_dict['train'] = temp_pretrain_cat_train_dict
            temp_ch_data_set_dict = temp_data_set_dict
        else:
            raise ValueError('Train Mode Error.')

        temp_data_loader_dict = {}
        num_data_set = len(temp_ch_data_set_dict)
        if len(batch_sz) != num_data_set:
            raise ValueError('Num of Data set != len(batch_sz)')

        for bsz_idx, data_set_name in enumerate(temp_ch_data_set_dict):
            temp_data_set_dict = temp_ch_data_set_dict[data_set_name]
            temp_data_set_loader = dataset_and_dataloader.CellDataLoader. \
                creat_data_loader(data_set_dict=temp_data_set_dict,
                                  batch_sz=batch_sz[bsz_idx],
                                  is_shuffle=is_shuffle,
                                  num_of_worker=num_of_worker)
            temp_data_loader_dict[data_set_name] = temp_data_set_loader

        return temp_data_loader_dict

    else:
        raise FileNotFoundError


def init_model(model: torch.nn.Module,
               model_param_dict: dict,
               device: torch.device = None) -> torch.nn.Module:
    """"""
    if model != {}:
        temp_model = model(**model_param_dict)
        temp_model = utility_function.init_model(temp_model)
        temp_model = temp_model.to(device)
        return temp_model
    else:
        raise ValueError('Model Parameter Dict Empty.')


def init_optimaizer_scheduler(inited_model: torch.nn.Module,
                              opt_param_dict: dict,
                              shc_param_dict: dict = None):
    """"""
    if (opt_param_dict != {}) and (shc_param_dict != {}):
        op_sch_dict = optim_sche.optimizer_select_and_init(inited_model,
                                                           opt_param_dict,
                                                           shc_param_dict)
        return op_sch_dict['optimizer'], op_sch_dict['scheduler']
    else:
        raise ValueError('Optimizer Parameter Dict or Scheduler Parameter Dict Empty.')


class Model_Run(object):
    """"""

    def __init__(self,
                 device: torch.device,
                 train_mode: str,
                 num_epoch: int,
                 data_loader,
                 preprocessor,
                 model: torch.nn.Module,
                 model_param: dict = None,
                 optimizer=None,
                 scheduler=None,
                 optimizer_param: dict = None,
                 scheduler_param: dict = None,
                 log_dir='.\\',
                 model_dir: str = None,
                 hyper_param: dict = None):
        """"""
        # store input
        self.device = device
        self.train_mode = train_mode
        self.num_epoch = num_epoch
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = model
        self.model_param = model_param
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_param = optimizer_param
        self.scheduler_param = scheduler_param
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.hyper_param = hyper_param

        # state var
        self.start_data_time = None
        self.current_model_name = None
        self.current_log_dir = None
        self.current_writer = None
        self.current_epoch_idx = None
        self.current_batch_idx = None
        self.current_model_idx = None
        self.current_set_num_batch = None
        self.current_batch_size = None
        self.current_lr = None
        self.current_num_model = None
        self.current_model_file_list = None
        self.current_model_dict = None
        self.current_data_loader = None
        self.current_stage = None

        # loss record
        self.current_epoch_train_loss = None
        self.current_epoch_test_loss = None
        self.current_batch_train_loss = None
        self.current_batch_test_loss = None
        self.current_batch_loss = None

        # output record
        self.current_epoch_train_output = None
        self.current_epoch_test_output = None
        self.current_batch_train_output = None
        self.current_batch_test_output = None
        self.current_batch_output = None

        # accumulator
        self.epoch_loss_accumulator = None
        self.epoch_pretrain_loss_accumulator = None

        # timer
        self.epoch_timer = None
        self.batch_train_timer = None
        self.batch_test_timer = None
        self.model_timer = None

    def run(self):
        """"""
        # record train start data time
        self.start_data_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # record current model name
        self.current_model_name = self.model.model_name

        # init tensorboard writer
        self.current_log_dir = os.path.join(self.log_dir,
                                            f'{self.current_model_name}_{self.start_data_time}')
        with SummaryWriter(log_dir=self.current_log_dir) as self.current_writer:
            if self.train_mode in ['pretrain', 'train', 'test']:
                self.current_model_idx = 0
                self.epoch_iter()
            elif self.train_mode == 'finetune':
                self.model_iter()
            else:
                raise ValueError('Train Mode Error')

    def model_iter(self):
        """"""

        self.finetune_init()

        for self.current_model_idx, file_name in enumerate(self.current_model_file_list):
            load_model_path = os.path.join(self.model_dir, file_name)
            self.current_model_dict = self.load_model(load_model_path)['model']
            temp_model = self.model.cpu()
            temp_model.set_model_attr(self.current_model_dict)
            self.model = temp_model.to(self.device)

            # init new optimizer and scheduler
            self.optimizer, self.scheduler = init_optimaizer_scheduler(self.model,
                                                                       self.optimizer_param,
                                                                       self.scheduler_param)

            self.epoch_iter()

    def epoch_iter(self):
        """"""
        # write hyper parameter
        if self.hyper_param is not None:
            self.current_writer.add_hparams(self.hyper_param,
                                            {'None': 0.0})

        # init epoch timer
        self.epoch_timer = utility_function.Timer()
        if self.train_mode == 'test':
            self.num_epoch = 1

        for self.current_epoch_idx in range(self.num_epoch):
            # start epoch timer
            self.epoch_timer.start()

            if self.train_mode == 'pretrain':
                self.current_epoch_train_loss = self.train_iter()
            elif self.train_mode in ['train', 'finetune']:
                self.current_epoch_train_loss = self.train_iter()
                self.current_epoch_test_loss = self.test_iter()
            elif self.train_mode == 'test':
                self.current_epoch_test_loss = self.test_iter()
            else:
                raise ValueError('Train Mode Error')

            # stop epoch timer
            self.epoch_timer.stop()

            # epoch logging
            self.logging_epoch()
            # write tensorboard
            self.write_res()
            if self.train_mode != 'test':
                # step scheduler
                self.scheduler.step()
                # save model
                self.save_model()

    def test_iter(self):
        """"""

        # init batch test timer and model timer
        self.batch_test_timer = utility_function.Timer()
        self.model_timer = utility_function.Timer()
        self.epoch_loss_accumulator = utility_function.Accumulator(1)
        self.current_stage = 'Test'

        # set model to eval mode
        self.model.eval()

        data_set_type = None
        if self.train_mode in ['train', 'finetune']:
            data_set_type = 'val'
        if self.train_mode == 'test':
            data_set_type = 'test'

        self.current_data_loader = self.data_loader[data_set_type]
        self.current_set_num_batch = len(self.data_loader[data_set_type])
        self.current_batch_size = self.data_loader[data_set_type].batch_size
        with torch.no_grad():
            for self.current_batch_idx, data_label in enumerate(self.data_loader[data_set_type]):
                # start batch test timer
                self.batch_test_timer.start()

                self.current_batch_test_loss = 0.0
                self.current_batch_test_loss, self.current_batch_test_output = self.batch_iter(data_label)

                # add batch test loss
                self.epoch_loss_accumulator.add(self.current_batch_train_loss)

                # stop batch test timer
                self.batch_test_timer.stop()

        return self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch

    def train_iter(self):
        """"""
        # init batch train timer and model timer
        self.batch_train_timer = utility_function.Timer()
        self.model_timer = utility_function.Timer()
        self.epoch_loss_accumulator = utility_function.Accumulator(1)

        if self.train_mode == 'pretrain':
            self.epoch_pretrain_loss_accumulator = utility_function.Accumulator(3)

        self.current_stage = 'Train'

        # set model to train mode
        self.model.train()

        data_set_type = self.train_mode
        if self.train_mode in ['finetune']:
            data_set_type = 'train'

        self.current_data_loader = self.data_loader[data_set_type]
        self.current_set_num_batch = len(self.current_data_loader)
        self.current_batch_size = self.current_data_loader.batch_size
        for self.current_batch_idx, data_label in enumerate(self.current_data_loader):
            # start batch train timer
            self.batch_train_timer.start()

            self.current_batch_train_loss = 0.0
            self.current_batch_train_loss, self.current_batch_train_output = self.batch_iter(data_label)

            # add batch train loss
            self.epoch_loss_accumulator.add(self.current_batch_train_loss)

            # stop batch train timer
            self.batch_train_timer.stop()

        return self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch

    def batch_iter(self, data_label: dict) -> torch.tensor:
        """"""

        # reset batch loss recorder
        self.current_batch_loss = 0.0

        if self.train_mode != 'test':
            # record current lr
            self.current_lr = self.optimizer.param_groups[0]['lr']

        # pre-process
        input_list = self.preprocessor.pro(data_label,
                                           self.train_mode,
                                           self.device)
        # start model timer
        self.model_timer.start()


        # run model
        model_loss = self.model(*input_list)

        # stop model timer
        self.model_timer.stop()

        self.current_batch_loss = model_loss.item()
        self.current_batch_output = None
        if self.train_mode != 'pretrain':
            self.current_batch_output = self.model.get_out()
            self.current_batch_output = self.current_batch_output.detach().cpu()

        if self.current_stage == 'Train':
            self.optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            self.optimizer.step()

        # batch logging
        self.logging_batch()

        if self.train_mode == 'pretrain':
            #[loss, MTP loss, NPP loss]
            self.epoch_pretrain_loss_accumulator.add(self.model.get_loss(),
                                                     self.model.get_mtp_loss(),
                                                     self.model.get_npp_loss())

        return self.current_batch_loss, self.current_batch_output

    def logging_batch(self, print_flag: bool = True):
        """"""
        separator = '--' * 80
        str_data = None
        if self.train_mode != 'test':
            str_data = '| Mode: {:s} | Epoch: {:3d}/{:3d} | Batch: {:5d}/{:5d} ' \
                       '| Lr: {:10.9f} | Batch Loss: {:8.7f} | Batch Time: {:5.2f} ms/batch |'.format(
                self.current_stage,
                self.current_epoch_idx + 1, self.num_epoch,
                self.current_batch_idx + 1, self.current_set_num_batch,
                self.current_lr,
                self.current_batch_loss,
                self.model_timer.times[-1] * 1000
            )

            if print_flag:
                print(separator)
                print(str_data)
                print(separator)

    def logging_epoch(self,
                      print_flag: bool = True,
                      write_flag: bool = True):
        """"""
        separator = '--' * 80
        str_data = None

        if self.train_mode == 'pretrain':
            str_data = '| Mode: {:s} | Epoch: {:3d}/{:3d} ' \
                       '| Lr: {:10.9f} | Epoch Loss: {:8.7f} | Epoch Time: {:5.2f} ms/batch |'.format(
                self.current_stage,
                self.current_epoch_idx + 1, self.num_epoch,
                self.current_lr,
                self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch,
                self.epoch_timer.times[-1] * 1000
            )
        elif self.train_mode in ['train', 'finetune']:
            pass
        else:
            pass

        if print_flag:
            print(separator)
            print(str_data)
            print(separator)

        if write_flag:
            write_file_name = f'{self.start_data_time}_' \
                              f'{self.current_model_name}_' \
                              f'{self.current_model_idx}_' \
                              f'{self.current_epoch_idx}_.txt'
            write_file_path = os.path.join(self.current_log_dir, write_file_name)
            utility_function.write_txt(write_file_path, separator)
            utility_function.write_txt(write_file_path, str_data)
            utility_function.write_txt(write_file_path, separator)

    def save_model(self):
        """"""
        model_save_path = os.path.join(self.current_log_dir, 'models')
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        save_name = f'{self.start_data_time}_' \
                    f'{self.current_model_name}_' \
                    f'{self.current_model_idx}_' \
                    f'{self.current_epoch_idx}_.pt'
        save_path = os.path.join(model_save_path, save_name)
        if not os.path.exists(save_path):
            model_to_save = self.model.get_save_model()
            save_dict = {
                'data_time': self.start_data_time,
                'model_name': self.current_model_name,
                'model_idx': self.current_model_idx,
                'epoch_idx': self.current_epoch_idx,
                'model': model_to_save,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler
            }
            torch.save(save_dict, save_path)
        else:
            raise FileExistsError('Current Model File Name Exist.')

    @staticmethod
    def load_model(model_path: str):
        """"""
        if os.path.exists(model_path):
            model_dict = torch.load(model_path)
            return model_dict
        else:
            raise FileNotFoundError(f'File No Find: {model_path}')

    def write_res(self):
        """"""
        if self.train_mode == 'pretrain':
            main_tag = f'{self.start_data_time}_{self.current_model_name}_{self.current_model_idx}'

            scalars_dict = {
                'Loss': self.epoch_pretrain_loss_accumulator[0] / self.current_set_num_batch,
                'MTP Loss': self.epoch_pretrain_loss_accumulator[1] / self.current_set_num_batch,
                'NPP Loss': self.epoch_pretrain_loss_accumulator[2] / self.current_set_num_batch,
            }

            self.current_writer.add_scalars(main_tag,
                                            scalars_dict,
                                            self.current_epoch_idx)

    def finetune_init(self):
        """"""
        if self.model_dir is None:
            raise ValueError('Need \'./models \' Path')

        if (self.optimizer_param is None) or (self.scheduler_param is None):
            raise ValueError('Need optimizer and scheduler parameter for finetune')

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError('Model File Path Not Find')

        self.current_model_file_list = os.listdir(self.model_dir)
        self.current_num_model = len(self.current_model_file_list)
