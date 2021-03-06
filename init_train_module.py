#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as skm

import optim_sche
import utility_function
import dataset_and_dataloader


def get_rnd_token_para(data_path, in_t_len, device):
    """"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data File: {data_path}  No Found')

    if in_t_len <= 0:
        raise ValueError('Input Token Len Must Greater than Zero')

    temp_data_set_dict = utility_function.read_pickle_file(data_path)

    cell_dict = {
        **temp_data_set_dict['pretrain'],
        **temp_data_set_dict['train'],
        **temp_data_set_dict['val'],
        **temp_data_set_dict['test'],
    }

    rnd_token_list = []
    rnd_para_list_dict = {
        'ch1v': [],
        'ch2v': [],
        'dcv': [],
        'ch3v': [],
        'ch3c': [],
    }

    with tqdm(total=len(cell_dict)) as bar:
        bar.set_description('Set rnd para and rnd token')
        for cell_name, cell_data in cell_dict.items():
            for para_name, para_data in cell_data.items():
                if para_name != 'label':
                    # get rnd para
                    temp_tensor_para_data = torch.from_numpy(para_data).to(torch.float32).unsqueeze(0)
                    rnd_para_list_dict[para_name].append(temp_tensor_para_data)

                    # get rnd token
                    temp_tensor_len = temp_tensor_para_data.shape[-1]
                    r = temp_tensor_len % in_t_len
                    if r != 0:
                        num_of_padding = in_t_len - r
                        pad_tensor = torch.zeros((1, num_of_padding), dtype=torch.float32)
                        temp_tensor_para_data = torch.cat((temp_tensor_para_data, pad_tensor), dim=-1)
                    tmp_rnd_token = temp_tensor_para_data.reshape(-1, in_t_len)
                    rnd_token_list.append(tmp_rnd_token)
            # update bar
            bar.update()

    rnd_token = torch.cat(rnd_token_list, dim=0)
    rnd_para = {}
    for key, val in rnd_para_list_dict.items():
        tmp_para_tensor = torch.cat(val, dim=0)
        rnd_para[key] = tmp_para_tensor

    return rnd_token, rnd_para



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
        elif train_mode in ['train', 'test']:
            temp_train_set = temp_data_set_dict['train']
            temp_pretrain_set = temp_data_set_dict['pretrain']
            temp_pretrain_cat_train_dict = {**temp_pretrain_set, **temp_train_set}
            del temp_data_set_dict['pretrain']
            temp_data_set_dict['train'] = temp_pretrain_cat_train_dict
            temp_ch_data_set_dict = temp_data_set_dict
        elif train_mode in ['finetune', ]:
            del temp_data_set_dict['pretrain']
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


        # print info
        print('-' * 64)
        print(f'Run Mode: {train_mode}')
        print('-' * 64)
        print(f'| data set | num of samples |')
        for key, val in temp_ch_data_set_dict.items():
            print(f'| {key} | {len(val)} |')
        print('-' * 64)

        print(f'| data set | batch size | num of batch |')
        for key, val in temp_data_loader_dict.items():
            print(f'| {key} | {val.batch_size} | {len(val)} |')
        print('-' * 64)

        return temp_data_loader_dict

    else:
        raise FileNotFoundError


def init_model(model: torch.nn.Module,
               model_param_dict: dict,
               device: torch.device = None) -> torch.nn.Module:
    """"""
    if model != {}:
        temp_model = model(**model_param_dict)
        temp_model = temp_model.init_model(model_param_dict)
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
                 data_loader,
                 preprocessor=None,
                 num_epoch: int = 1,
                 model: torch.nn.Module = None,
                 loss_fn_list= None,
                 model_param: dict = None,
                 optimizer=None,
                 scheduler=None,
                 optimizer_param: dict = None,
                 scheduler_param: dict = None,
                 log_dir='.\\',
                 model_dir: str = None,
                 hyper_param: dict = None,
                 num_class: int = None,
                 pretrain_model_name: str = None,
                 pretrain_step: int = 16):
        """"""
        # store input
        self.device = device
        self.train_mode = train_mode
        self.num_epoch = num_epoch
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = model
        self.loss_fn_list = loss_fn_list
        self.model_param = model_param
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_param = optimizer_param
        self.scheduler_param = scheduler_param
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.hyper_param = hyper_param
        self.num_class = num_class
        self.pretrain_model_name = pretrain_model_name
        self.pretrain_step = pretrain_step

        # state var
        self.start_data_time = None
        self.current_model_name = None
        self.current_log_dir = None
        self.current_writer = None
        self.current_epoch_idx = None
        self.current_batch_idx = None
        self.current_model_idx = None
        self.current_model_epoch_idx = None
        self.current_set_num_batch = None
        self.current_batch_size = None
        self.current_lr = None
        self.num_model_idx = None
        self.num_model_epoch = None
        self.current_model_list = None
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
        self.current_epoch_eval_dict = None
        self.train_epoch_eval_dict = None
        self.test_epoch_eval_dict = None

        # output record
        self.current_epoch_output_label_list = None
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
        if self.pretrain_model_name:
            self.current_model_name = self.pretrain_model_name
        elif self.model:
            if self.model.model_name:
                self.current_model_name = self.model.model_name
            else:
                raise ValueError('Model Name Do Not Exist.')
        else:
            raise ValueError('No Model or Model Name.')

        # init tensorboard writer
        self.current_log_dir = os.path.join(self.log_dir,
                                            f'{self.train_mode}_{self.current_model_name}_{self.start_data_time}')
        with SummaryWriter(log_dir=self.current_log_dir) as self.current_writer:
            run_out = None
            if self.train_mode in ['pretrain', 'train']:
                self.current_model_epoch_idx = 0
                self.num_model_epoch = 1
                run_out = self.epoch_iter()
            elif self.train_mode == 'finetune':
                run_out = self.model_iter()
            elif self.train_mode == 'test':
                run_out = self.model_iter()
            else:
                raise ValueError('Train Mode Error')

        return run_out

    def model_iter(self):
        """"""

        self.finetune_init()
        epoch_out_list = []

        for self.current_model_idx, model_idx in enumerate(self.current_model_list):
            temp_model_path = os.path.join(self.model_dir, model_idx)
            self.current_model_file_list = os.listdir(temp_model_path)
            self.num_model_epoch = len(self.current_model_file_list)

            if self.pretrain_step:
                temp_model_epoch_file_list = []
                mod_res = self.num_model_epoch % self.pretrain_step
                if mod_res == 0:
                    temp_num_model_epoch = self.num_model_epoch

                else:
                    temp_num_model_epoch = self.num_model_epoch - mod_res

                file_idx_range = range(0, temp_num_model_epoch, self.pretrain_step)
                temp_model_epoch_file_list = [self.current_model_file_list[idx] for idx in file_idx_range]
                self.current_model_file_list = temp_model_epoch_file_list

            for self.current_model_epoch_idx, file_name in enumerate(self.current_model_file_list):
                load_model_path = os.path.join(temp_model_path, file_name)
                temp_load_dict = self.load_model(load_model_path)
                self.current_model_dict = temp_load_dict['model']

                if self.train_mode in ['finetune', 'test']:
                    split_file_name = file_name.split('_')
                    temp_current_model_idx = split_file_name[2]
                    self.current_model_idx = int(temp_current_model_idx.split('-')[-1])
                    temp_current_model_epoch_idx = split_file_name[3]
                    self.current_model_epoch_idx = int(temp_current_model_epoch_idx.split('-')[-1])

                self.hyper_param = None
                if 'hyper_param' in temp_load_dict.keys():
                    self.hyper_param = temp_load_dict['hyper_param']

                temp_model = self.model.cpu()
                temp_model.set_model_attr(self.current_model_dict)

                # init model
                self.model = None
                self.model = temp_model.to(self.device)

                if self.train_mode == 'finetune':
                    # init new optimizer and scheduler
                    self.optimizer = None
                    self.scheduler = None
                    self.optimizer, self.scheduler = init_optimaizer_scheduler(self.model,
                                                                               self.optimizer_param,
                                                                               self.scheduler_param)
                epoch_out = self.epoch_iter()
                epoch_out_list.append(epoch_out)

        if not epoch_out_list:
            raise ValueError(f'{self.train_mode} Model iter Out: Empty Out.')
        else:
            return max(epoch_out_list)



    def epoch_iter(self):
        """"""
        # write hyper parameter
        if self.hyper_param is not None:
            self.current_writer.add_hparams(self.hyper_param,
                                            {'None': 0.0})

        tmp_epoch_loss_accumulator = utility_function.Accumulator(1)
        # init epoch timer
        self.epoch_timer = utility_function.Timer()

        for self.current_epoch_idx in range(self.num_epoch):
            # start epoch timer
            self.epoch_timer.start()

            if self.train_mode == 'pretrain':
                self.current_epoch_train_loss = self.train_iter()
                tmp_epoch_loss_accumulator.add(self.current_epoch_train_loss)
            elif self.train_mode in ['train', 'finetune']:
                self.current_epoch_train_loss = self.train_iter()
                self.current_epoch_test_loss = self.test_iter()
                tmp_epoch_loss_accumulator.add(self.current_epoch_test_loss)
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

        if self.train_mode == 'pretrain':
            if not self.current_epoch_train_loss:
                raise ValueError(f'{self.train_mode} Epoch Out: None or Empty Loss.')
            return tmp_epoch_loss_accumulator.data[-1] / self.num_epoch
        elif self.model.model_name == 'CAE':
            return tmp_epoch_loss_accumulator.data[-1] / self.num_epoch
        else:
            if not self.test_epoch_eval_dict:
                raise ValueError(f'{self.train_mode} Epoch Out: None or Empty Eval Dict.')
            return self.test_epoch_eval_dict['test_top_1_acc']

    def test_iter(self):
        """"""

        # init batch test timer and model timer
        self.batch_test_timer = utility_function.Timer()
        self.model_timer = utility_function.Timer()
        self.epoch_loss_accumulator = utility_function.Accumulator(1)
        self.current_epoch_output_label_list = []
        self.current_epoch_eval_dict = {}
        self.test_epoch_eval_dict = {}

        self.current_stage = 'Test'

        # set model to eval mode
        self.model.eval()

        data_set_type = None
        if self.train_mode in ['train', 'finetune']:
            data_set_type = 'val'
        elif self.train_mode == 'test':
            data_set_type = 'test'
        else:
            pass

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
                self.epoch_loss_accumulator.add(self.current_batch_test_loss)

                # stop batch test timer
                self.batch_test_timer.stop()

        # calculate epoch eval
        if (self.train_mode != 'pretrain') and (self.model.model_name != 'CAE'):
            self.current_epoch_eval_dict = self.cal_accuracy(self.current_epoch_output_label_list)
            for key, val in self.current_epoch_eval_dict.items():
                tmp_key = 'test_' + key
                self.test_epoch_eval_dict[tmp_key] = copy.deepcopy(val)

        return self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch

    def train_iter(self):
        """"""
        # init batch train timer and model timer
        self.batch_train_timer = utility_function.Timer()
        self.model_timer = utility_function.Timer()
        self.epoch_loss_accumulator = utility_function.Accumulator(1)
        self.current_epoch_output_label_list = []
        self.current_epoch_eval_dict = {}
        self.train_epoch_eval_dict = {}

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
            self.current_batch_train_output = None
            self.current_batch_train_loss, self.current_batch_train_output = self.batch_iter(data_label)

            # add batch train loss
            self.epoch_loss_accumulator.add(self.current_batch_train_loss)

            # stop batch train timer
            self.batch_train_timer.stop()

        # calculate epoch eval
        if (self.train_mode != 'pretrain') and (self.model.model_name != 'CAE'):
            self.current_epoch_eval_dict = self.cal_accuracy(self.current_epoch_output_label_list)
            for key, val in self.current_epoch_eval_dict.items():
                tmp_key = 'train_' + key
                self.train_epoch_eval_dict[tmp_key] = copy.deepcopy(val)

        return self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch

    def batch_iter(self, data_label: dict) -> torch.tensor:
        """"""

        # reset batch loss recorder
        self.current_batch_loss = 0.0

        if self.train_mode != 'test':
            # record current lr
            self.current_lr = self.optimizer.param_groups[0]['lr']

        # pre-process
        input_tulpe = self.preprocessor.pro(data_label,
                                           self.train_mode,
                                           self.device)
        # start model timer
        self.model_timer.start()

        # run model
        # data_dict, label_one_hot, rpl_label_onehot, train_mode
        model_out_tulpe = self.model(*input_tulpe)

        # stop model timer
        self.model_timer.stop()

        model_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        model_out = None
        if self.train_mode == 'pretrain':
            model_out, _, NSP_pred, NSP_label, MLM_pred, MLM_label = model_out_tulpe
            NPP_Loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            MTP_Loss = torch.zeros(1, dtype=torch.float32, device=self.device)

            if NSP_pred is not None:
                NPP_Loss = self.loss_fn_list[0](NSP_pred.to(torch.float32),
                                                NSP_label.to(torch.float32))
            if MLM_pred is not None:
                MTP_Loss = self.loss_fn_list[1](MLM_pred.to(torch.float32),
                                                MLM_label.to(torch.float32))
            model_loss =  MTP_Loss + NPP_Loss

            # [loss, NPP loss, MTP loss]
            self.epoch_pretrain_loss_accumulator.add(model_loss.item(),
                                                     NPP_Loss.item(),
                                                     MTP_Loss.item())
        else:
            model_out = model_out_tulpe[0]
            out_data = model_out_tulpe[0]
            label = model_out_tulpe[1]
            if self.train_mode != 'test':
                if len(self.loss_fn_list) != 1:
                    raise ValueError(f'More Than One Loss Func For {self.train_mode}.')
                model_loss = self.loss_fn_list[0](out_data.to(torch.float32),
                                                  label.to(torch.float32))

            self.current_epoch_output_label_list.append((model_out, label))

        if self.current_stage == 'Train':
            if self.current_batch_loss is not None:
                self.optimizer.zero_grad()
                if self.train_mode == 'pretrain':
                    model_loss.backward(retain_graph=True)
                else:
                    model_loss.backward()
                self.optimizer.step()

        self.current_batch_loss = model_loss.item()
        if model_out is not None:
            self.current_batch_output = model_out

        # batch logging
        self.logging_batch()

        return self.current_batch_loss, self.current_batch_output

    def cal_accuracy(self, out_label_list):
        """"""
        tmp_out_list = []
        tmp_label_list = []

        if not out_label_list:
            raise ValueError('Output and Label List is Empty.')

        for o, l in out_label_list:
            tmp_out_list.append(o)
            tmp_label_list.append(l)

        if len(tmp_out_list) != len(tmp_label_list):
            raise ValueError('Number of Output != Number of Label')

        tmp_out = torch.cat(tmp_out_list)
        tmp_label = torch.cat(tmp_label_list)

        if tmp_out.shape != tmp_label.shape:
            raise ValueError('Shape of Output != Shape of Label')


        tmp_out_onehot_arr = tmp_out.detach().cpu().numpy()
        tmp_out_arr = torch.argmax(tmp_out, dim=1).detach().cpu().numpy()

        tmp_label_onehot_arr = tmp_label.detach().cpu().numpy()
        tmp_label_arr = torch.argmax(tmp_label, dim=1).detach().cpu().numpy()

        label_list = [x for x in range(self.num_class)]
        label_arr = np.array(label_list)

        top_1_acc = skm.top_k_accuracy_score(y_true=tmp_label_arr, y_score=tmp_out_onehot_arr, k=1, labels=label_arr)
        top_3_acc = skm.top_k_accuracy_score(y_true=tmp_label_arr, y_score=tmp_out_onehot_arr, k=3, labels=label_arr)

        f1_mac_score = skm.f1_score(y_true=tmp_label_arr, y_pred=tmp_out_arr,
                                    labels=label_arr, average='macro')
        f1_mic_score = skm.f1_score(y_true=tmp_label_arr, y_pred=tmp_out_arr,
                                    labels=label_arr, average='micro')

        precision_mac = skm.precision_score(y_true=tmp_label_arr, y_pred=tmp_out_arr,
                                            labels=label_arr, average='macro')
        precision_mic = skm.precision_score(y_true=tmp_label_arr, y_pred=tmp_out_arr,
                                            labels=label_arr, average='micro')

        recall_mac = skm.recall_score(y_true=tmp_label_arr, y_pred=tmp_out_arr,
                                      labels=label_arr, average='macro')
        recall_mic = skm.recall_score(y_true=tmp_label_arr, y_pred=tmp_out_arr,
                                      labels=label_arr, average='micro')

        roc_auc_mac = skm.roc_auc_score(y_true=tmp_label_onehot_arr,
                                        y_score=tmp_out_onehot_arr,
                                        average='macro',
                                        multi_class='ovr',
                                        labels=label_arr)
        roc_auc_mic = skm.roc_auc_score(y_true=tmp_label_onehot_arr,
                                        y_score=tmp_out_onehot_arr,
                                        average='micro',
                                        multi_class='ovr',
                                        labels=label_arr)

        tmp_eval_dict = {
            'top_1_acc': top_1_acc,
            'top_3_acc': top_3_acc,
            'f1_mac_score': f1_mac_score,
            'f1_mic_score': f1_mic_score,
            'precision_mac': precision_mac,
            'precision_mic': precision_mic,
            'recall_mac': recall_mac,
            'recall_mic': recall_mic,
            'roc_auc_mac':roc_auc_mac,
            'roc_auc_mic': roc_auc_mic
        }

        return tmp_eval_dict

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
            str_data = '| Mode: {:s} | Models: {:3d}/{:3d} |Epoch: {:3d}/{:3d} ' \
                       '| Lr: {:10.9f} | Epoch Loss: {:8.7f} | Epoch Time: {:5.2f} s/batch |'.format(
                self.train_mode,
                self.current_model_epoch_idx + 1, self.num_model_epoch,
                self.current_epoch_idx + 1, self.num_epoch,
                self.current_lr,
                self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch,
                self.epoch_timer.times[-1]
            )
        elif self.train_mode in ['finetune',]:
            str_data = '| Mode: {:s} | Models: {:3d}/{:3d} | Epoch: {:3d}/{:3d} ' \
                       '| Epoch Loss: {:8.7f} | Epoch Time: {:5.2f} s/batch |'.format(
                self.train_mode,
                self.current_model_epoch_idx + 1, self.num_model_epoch,
                self.current_epoch_idx + 1, self.num_epoch,
                self.epoch_loss_accumulator.data[-1] / self.current_set_num_batch,
                self.epoch_timer.times[-1]
            )
        elif self.train_mode in ['test', ]:
            str_data = '| Mode: {:s} | Models: {:3d}/{:3d} | Epoch: {:3d}/{:3d} ' \
                       '| Epoch Time: {:5.2f} s/batch |'.format(
                self.train_mode,
                self.current_model_idx + 1, self.num_model_idx,
                self.current_model_epoch_idx + 1, self.num_model_epoch,
                self.epoch_timer.times[-1]
            )
        else:
            str_data = '| Mode: {:s} | Models: {:3d}/{:3d} | Epoch: {:3d}/{:3d} ' \
                       '| Epoch Time: {:5.2f} s/batch |'.format(
                self.train_mode,
                self.current_model_epoch_idx + 1, self.num_model_epoch,
                self.current_model_epoch_idx + 1, self.num_model_epoch,
                self.epoch_timer.times[-1]
            )

        if print_flag:
            print(separator)
            print(str_data)
            print(separator)

        if write_flag:

            if self.train_mode in ['pretrain', ]:
                write_file_name = f'{self.train_mode}_' \
                                  f'{self.start_data_time}_' \
                                  f'{self.current_model_name}_' \
                                  f'{self.current_model_epoch_idx}.txt'
            elif self.train_mode in ['test', ]:
                write_file_name = f'{self.train_mode}_' \
                                  f'{self.start_data_time}_' \
                                  f'{self.current_model_name}_' \
                                  f'{self.current_model_idx}.txt'
            elif self.train_mode in ['finetune', ]:
                if not self.model_dir:
                    raise FileNotFoundError('Model Path For Finetune Do Not Exist.')
                fine_tune_name = self.model_dir.split('\\')[2]
                write_file_name = f'{fine_tune_name}_' \
                                  f'{self.train_mode}_' \
                                  f'{self.start_data_time}_' \
                                  f'{self.current_model_name}_' \
                                  f'{self.current_model_idx}.txt'
            else:
                write_file_name = f'{self.train_mode}_' \
                                  f'{self.start_data_time}_' \
                                  f'{self.current_model_name}_' \
                                  f'{self.current_model_epoch_idx}.txt'

            write_file_path = os.path.join(self.current_log_dir, write_file_name)
            utility_function.write_txt(write_file_path, separator)
            utility_function.write_txt(write_file_path, str_data)
            utility_function.write_txt(write_file_path, separator)

    def save_model(self):
        """"""
        model_save_path = os.path.join(self.current_log_dir, f'models\\{self.current_model_epoch_idx}')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        save_name = f'{self.start_data_time}_' \
                    f'{self.current_model_name}_' \
                    f'mid-{self.current_model_epoch_idx}_' \
                    f'eid-{self.current_epoch_idx}_.pt'
        save_path = os.path.join(model_save_path, save_name)
        if not os.path.exists(save_path):
            model_to_save = self.model.get_save_model()
            save_dict = {
                'data_time': self.start_data_time,
                'model_name': self.current_model_name,
                'model_idx': self.current_model_epoch_idx,
                'epoch_idx': self.current_epoch_idx,
                'model': model_to_save,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler
            }
            if self.hyper_param:
                save_dict.update({'hyper_param': self.hyper_param})
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
            main_tag = f'{self.train_mode}_{self.start_data_time}_{self.current_model_name}_{self.current_model_epoch_idx}'

            scalars_dict = {
                'Loss': self.epoch_pretrain_loss_accumulator[0] / self.current_set_num_batch,
                'MTP Loss': self.epoch_pretrain_loss_accumulator[1] / self.current_set_num_batch,
                'NPP Loss': self.epoch_pretrain_loss_accumulator[2] / self.current_set_num_batch,
            }
            self.current_writer.add_scalars(main_tag,
                                            scalars_dict,
                                            self.current_epoch_idx)

        elif self.train_mode in ['finetune', 'train']:
            main_tag = f'{self.train_mode}_{self.start_data_time}_{self.current_model_name}_{self.current_model_epoch_idx}'

            if self.model.model_name != 'CAE':
                scalars_dict_train = self.train_epoch_eval_dict
                scalars_dict_val = self.test_epoch_eval_dict

                self.current_writer.add_scalars(main_tag,
                                                scalars_dict_train,
                                                self.current_epoch_idx)
                self.current_writer.add_scalars(main_tag,
                                                scalars_dict_val,
                                                self.current_epoch_idx)
            else:
                scalars_dict = {
                    'Train Loss': self.current_epoch_train_loss,
                    'Val Loss': self.current_epoch_test_loss,
                }
                self.current_writer.add_scalars(main_tag,
                                                scalars_dict,
                                                self.current_epoch_idx)
        elif self.train_mode == 'test':
            main_tag = f'{self.train_mode}_{self.start_data_time}_{self.current_model_name}_{self.current_model_idx}'
            scalars_dict_test = self.test_epoch_eval_dict
            self.current_writer.add_scalars(main_tag,
                                            scalars_dict_test,
                                            self.current_model_epoch_idx)

        else:
            pass



    def finetune_init(self):
        """"""
        if self.model_dir is None:
            raise ValueError('Need \'./models \' Path')

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError('Model File Path Not Find')

        self.current_model_list = os.listdir(self.model_dir)
        self.num_model_idx = len(self.current_model_list)

        if self.train_mode == 'test':
            if self.num_epoch > 1:
                raise ValueError('Epoch > 1 For Test.')

        if self.train_mode == 'finetune':
            if (self.optimizer_param is None) or (self.scheduler_param is None):
                raise ValueError('Need optimizer and scheduler parameter for finetune')

            if self.num_model_idx > 1:
                raise ValueError('Too Much Model for finetune.')

