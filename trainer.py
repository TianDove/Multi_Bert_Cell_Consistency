#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
from datetime import datetime

#
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as skm
#
import opt
import utility_function


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n  # 创建一个有n个元素的列表

    def add(self, *args):  # 将*arg是按顺序加到对应的self.data的位置上
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

    def reset(self):
        """清空计时器"""
        self.times = []


class CellDataLoader(Dataset):
    """"""
    def __init__(self,
                 data_dict: dict,
                 type_data_set: str) -> None:
        """"""
        self.current_data_set = data_dict[type_data_set]
        self.current_data_list = list(self.current_data_set.values())
        self.n_samples = None

    def __getitem__(self, index: int) -> tuple:
        """"""
        return self.current_data_list[index]

    def __len__(self) -> int:
        """"""
        self.n_samples = len(self.current_data_set)
        return self.n_samples

    @classmethod
    def creat_data_loader(cls,
                          data_dict: dict,
                          type_data_set: str,
                          batch_sz: int,
                          num_of_worker: int) -> DataLoader:
        """"""
        data_set = cls(data_dict, type_data_set)
        data_loader = DataLoader(data_set,
                                 batch_size=batch_sz,
                                 shuffle=True,
                                 num_workers=num_of_worker,
                                 pin_memory=True,
                                 drop_last=True)
        return data_loader


class Trainer(object):
    """"""
    def __init__(self,
                 device_type: str,
                 log_dir: str,
                 data_path: str,
                 batch_size: int,
                 workers: int,
                 model,
                 model_param: dict,
                 optimizer_param: dict,
                 scheduler_param: dict,
                 num_cls: int = 8):
        """"""
        self.current_data_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # input
        self.device = self.device_init(device_type)
        self.log_dir = log_dir
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.model = model
        self.model_param = model_param
        self.optimizer_param = optimizer_param
        self.scheduler_param = scheduler_param
        self.num_cls = num_cls
        self.model_name = None

        # model relate
        self.data_loader_dict = None
        self.init_model = None
        self.preprocess = None
        self.optm = None
        self.sche = None
        self.num_epoch = None
        self.num_batch = None
        self.writer_dir = None

        # accumulator
        self.epoch_train_loss_accumulator = Accumulator(1)
        self.batch_train_loss_accumulator = Accumulator(1)
        self.epoch_val_loss_accumulator = Accumulator(1)
        self.batch_nsp_loss_accumulator = Accumulator(1)
        self.batch_mlm_loss_accumulator = Accumulator(1)

        # model out list
        self.epoch_train_out_label_list = []
        self.epoch_val_out_label_list = []

        # state
        self.current_model_name = self.model.__class__.__name__
        self.current_writer = None
        self.curr_epoch = None
        self.curr_batch = None
        self.curr_stage = None
        self.curr_lr = None
        self.curr_batch_loss = None
        self.curr_batch_size = None

        # loss
        self.epoch_train_loss = 0.0
        self.batch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.test_loss = 0.0

        # timer
        self.epoch_timer = Timer()
        self.batch_train_timer = Timer()
        self.model_timer = Timer()
        self.epoch_val_timer = Timer()
        self.test_timer = Timer()

        # eval
        self.epoch_train_acc = 0.0
        self.epoch_val_acc = 0.0
        self.test_acc = 0.0

    @staticmethod
    def device_init(device_type: str) -> torch.device:
        """"""
        device = None
        if device_type == 'gpu':
            device = utility_function.try_gpu()
        elif device_type == 'cpu':
            device = torch.device('cpu')
        else:
            raise ValueError('Device Type Error.')
        return device

    @staticmethod
    def read_datafile(data_path: str) -> dict:
        """
        Read data file.
        data = {
            'pretrain': dict{cell_para_dict},
            'train': dict{cell_para_dict},
            'valid': dict{cell_para_dict},
            'test': dict{cell_para_dict},
            }
        """
        assert os.path.exists(data_path), 'Data File Not Found.'

        data_set_dict = utility_function.read_pickle_file(data_path)

        return data_set_dict

    def data_loader_init(self,
                         data_set_dict: dict,
                         batch_size: int,
                         workers: int) -> dict:
        """"""
        if self.curr_stage == 'pretrain':
            data_set_type = ('pretrain', 'train', 'val', 'test')
        else:
            data_set_type = ('train', 'val', 'test')

        for key in data_set_type:
            assert key in data_set_dict.keys()

        batch_size_list = [batch_size] * len(data_set_type)  # [pretrain, train, val, test]

        temp_data_loader_dict = {}
        for tp, bsz in iter(zip(data_set_type, batch_size_list)):
            temp_data_loader_dict[tp] = CellDataLoader.creat_data_loader(data_dict=data_set_dict,
                                                                         type_data_set=tp,
                                                                         batch_sz=bsz,
                                                                         num_of_worker=workers)
        return temp_data_loader_dict

    @staticmethod
    def model_init(model,
                   model_param: dict,
                   device: torch.device):
        """"""
        assert model_param != {}

        temp_model = model(**model_param)
        temp_model = utility_function.init_model(temp_model)
        temp_model = temp_model.to(device)

        return temp_model

    @staticmethod
    def opt_sch_init(model,
                     optimizer_param: dict,
                     scheduler_param: dict) -> (torch.optim.Optimizer, torch.optim.lr_scheduler):
        """"""
        assert optimizer_param != {}
        assert scheduler_param != {}

        op_sch_dict = opt.optimizer_select_and_init(model,
                                                    optimizer_param,
                                                    scheduler_param)
        return op_sch_dict['optimizer'], op_sch_dict['scheduler']

    def train_iter(self,
                   train_mode: str):
        """"""
        self.curr_stage = train_mode

        # set model to train mode
        self.init_model.train()

        # local accumulator
        self.num_batch = self.data_loader_dict[self.curr_stage].__len__()
        self.curr_batch_size = self.data_loader_dict[self.curr_stage].batch_size
        self.batch_train_loss_accumulator.reset()

        self.batch_nsp_loss_accumulator.reset()
        self.batch_mlm_loss_accumulator.reset()

        if self.curr_stage == 'train':
            self.batch_train_timer.reset()

        for batch_index, train_data in enumerate(self.data_loader_dict[self.curr_stage]):
            # train_data
            # train_data = {
            #     'ch1v': tensor(batch_size, param_len),
            #      ...
            # }

            # batch train timer start
            self.batch_train_timer.start()

            # log current batch index
            self.curr_batch = batch_index

            temp_pre_list = self.preprocess.pro(train_data,
                                                self.device)

            batch_loss = self.init_model(*temp_pre_list)

            self.optm.zero_grad()
            batch_loss.backward()
            self.optm.step()

            # batch_train timer end
            self.batch_train_timer.stop()

            # log batch loss
            # batch_loss_list.append(model_loss.to('cpu').item())
            r_loss = batch_loss.cpu().item()
            self.curr_batch_loss = r_loss
            self.batch_train_loss_accumulator.add(r_loss)

            # accumulate nsp mlm loss
            if self.curr_stage == 'pretrain':
                current_nsp_loss = self.init_model.get_nsp_loss().detach().cpu().item()
                current_mlm_loss = self.init_model.get_mlm_loss().detach().cpu().item()

                self.batch_nsp_loss_accumulator.add(current_nsp_loss)
                self.batch_mlm_loss_accumulator.add(current_mlm_loss)

            #  batch accuracy
            if self.curr_stage != 'pretrain':
                self.epoch_train_out_label_list.append((self.init_model.get_out().detach().cpu(),
                                                       temp_pre_list[1].detach().cpu()))

            # log other
            self.log_func()

        return r_loss

    def test_iter(self,
                  test_mode: str):
        """"""
        self.curr_stage = test_mode

        # set model to eval mode
        self.init_model.eval()
        self.num_batch = self.data_loader_dict[self.curr_stage].__len__()
        self.curr_batch_size = self.data_loader_dict[self.curr_stage].batch_size

        with torch.no_grad():
            for batch_idx, train_data in enumerate(self.data_loader_dict[self.curr_stage]):
                # log current batch index
                self.curr_batch = batch_idx

                temp_pre_list = self.preprocess.pro(train_data,
                                                    self.device)

                if self.curr_stage == 'test':
                    self.model_timer.start()

                self.curr_batch_loss = self.init_model(*temp_pre_list)

                if self.curr_stage == 'test':
                    self.model_timer.stop()

                r_loss = self.curr_batch_loss.to('cpu').item()
                self.epoch_val_out_label_list.append((self.init_model.get_out().detach().cpu(),
                                                      temp_pre_list[1].detach().cpu()))

        return r_loss

    def save_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        save_model_path = os.path.join(path, 'models')
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        file_name = f'{self.model_name}_{self.current_data_str}_{self.curr_epoch}'
        data_type = '.pt'
        save_path = os.path.join(save_model_path, file_name + data_type)
        if self.curr_stage == 'pretrain':
            save_model_dict = {
                'tokensub': self.init_model.token_substitution.sp_token_embedding,
                'encoder': self.init_model.encoder,
            }
        else:
            save_model_dict = self.init_model
        torch.save({
            'epoch': self.curr_epoch,
            'model': save_model_dict,
            'optimizer': self.optm,
            'scheduler': self.sche,
        }, save_path)

    def load_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        temp_dict = torch.load(path)
        self.init_model = temp_dict['model']
        self.optm = temp_dict['optimizer']
        self.sche = temp_dict['scheduler']

    def log_func(self):
        """"""
        separator = '-' * 120
        str_data = None
        if self.curr_stage in ['pretrain', 'train']:
            str_data = '| Epoch {:3d}/{:3d} | Batch: {:5d}/{:5d} ' \
                       '| lr {:10.9f} | {:5.2f} ms/batch  | ' \
                       'Train Loss {:8.7f} |'.format(
                        self.curr_epoch, self.num_epoch,
                        self.curr_batch, self.num_batch,
                        self.curr_lr, self.batch_train_timer.times[-1] * 1000,
                        self.curr_batch_loss)

        elif self.curr_stage == 'val':
            str_data = '| End of epoch {:3d} | Current Learning Rat: {:8.7f} | ' \
                       '| Train Accuracy {:5.4f} | Valid Accuracy {:5.4f}  |'.format(
                        self.curr_epoch, self.curr_lr,
                        self.epoch_train_acc, self.epoch_val_acc)

        elif self.curr_stage == 'test':
            str_data = '| Test Stage | Test Total Time: {:5.2f} ms | ' \
                       'Test Avg Loss: {:8.7f} |Test Accuracy {:5.4f} | '.format(
                        self.test_timer.sum() * 1000,
                        self.test_loss, self.test_acc)
        else:
            raise ValueError('Log Type Error.')

        # show log
        print(separator)
        print(str_data)
        print(separator)

        if self.curr_stage in ['val', 'test']:
            utility_function.write_txt(os.path.join(self.writer_dir, 'print_log.txt'), separator)
            utility_function.write_txt(os.path.join(self.writer_dir, 'print_log.txt'), str_data)
            utility_function.write_txt(os.path.join(self.writer_dir, 'print_log.txt'), separator)

    def writer_eval(self):
        """"""
        eval_dict = {}
        if self.curr_stage == 'pretrain':
            eval_dict = {
                'pretrain-nsp_loss': self.batch_nsp_loss_accumulator[-1] / self.num_batch,
                'pretrain-mlm_loss': self.batch_mlm_loss_accumulator[-1] / self.num_batch,
            }
        else:
            epoch_train_out_arr, epoch_train_label_arr = self._unpack_to_array(self.epoch_train_out_label_list)
            epoch_train_top_1_acc = self.accuracy_batch(epoch_train_out_arr, epoch_train_label_arr)
            self.epoch_train_acc = epoch_train_top_1_acc

            epoch_val_out_arr, epoch_val_label_arr = self._unpack_to_array(self.epoch_val_out_label_list)
            epoch_val_top_1_acc = self.accuracy_batch(epoch_val_out_arr, epoch_val_label_arr)
            self.epoch_val_acc = epoch_val_top_1_acc

            eval_dict = {
                'epoch_train_top_1_acc': epoch_train_top_1_acc,
                'epoch_val_top_1_acc': epoch_val_top_1_acc,
            }

        self.current_writer.add_scalars(f'{self.model_name}_{self.current_data_str}',
                                        eval_dict, self.curr_epoch)

    @staticmethod
    def _unpack_to_array(out_label_list):
        """"""
        out_list = []
        label_list = []
        for out, label in iter(out_label_list):
            out_list.append(out)
            label_list.append(label)

        out_arr = torch.cat(out_list)
        label_arr = torch.cat(label_list)

        return out_arr, label_arr

    def accuracy_batch(self, out, label):
        """"""
        label_idx = np.arange(self.num_cls)
        temp_shape = out.shape
        label_idx_tensor = torch.argmax(label, dim=1).detach().cpu().numpy()
        out_tensor = out.detach().cpu().numpy()
        out_idx_tensor = torch.argmax(out, dim=1).detach().cpu().numpy()

        top_1_acc = skm.top_k_accuracy_score(label_idx_tensor, out_tensor, k=1, labels=label_idx)
        return top_1_acc

    def run(self,
            train_mode: str,
            num_epoch: int,
            preprocessor):
        """"""
        self.curr_stage = train_mode

        # read data file
        data_set_dict = self.read_datafile(self.data_path)
        # init data loader
        self.data_loader_dict = self.data_loader_init(data_set_dict,
                                                      self.batch_size,
                                                      self.workers)
        # init model
        self.init_model = self.model_init(self.model,
                                          self.model_param,
                                          self.device)
        self.model_name = self.init_model.__class__.__name__
        # init optimizer
        self.optm, self.sche = self.opt_sch_init(self.init_model,
                                                 self.optimizer_param,
                                                 self.scheduler_param)
        # init preprocess fun
        self.preprocess = preprocessor
        # start run
        self.writer_dir = os.path.join(self.log_dir,
                                      f'{self.init_model.model_name}_{self.current_data_str}')

        with SummaryWriter(log_dir=self.writer_dir) as self.current_writer:

            self.num_epoch = num_epoch
            for epoch in range(num_epoch):
                # log current epoch index
                self.curr_epoch = epoch

                # log current lr
                self.curr_lr = self.optm.param_groups[0]['lr']

                # epoch timer start
                self.epoch_timer.start()
                self.epoch_train_loss_accumulator.reset()
                self.epoch_train_out_label_list = []

                # clear loss
                self.epoch_train_loss = 0.0

                # train
                self.epoch_train_loss = self.train_iter(train_mode)

                # log epoch train loss
                self.epoch_train_loss_accumulator.add(self.epoch_train_loss)

                # step scheduler
                if self.sche is not None:
                    self.sche.step()

                if self.curr_stage != 'pretrain':
                    # clear val loss
                    self.epoch_val_loss = 0.0
                    # epoch val timer start
                    self.epoch_val_timer.start()
                    self.epoch_val_loss_accumulator.reset()
                    self.epoch_val_out_label_list = []

                    # val
                    self.epoch_val_loss = self.test_iter('val')

                    self.epoch_val_loss_accumulator.add(self.epoch_val_loss)

                    # epoch val timer end
                    self.epoch_val_timer.stop()

                # epoch timer end
                self.epoch_timer.stop()

                # write eval
                self.writer_eval()
                self.log_func()
                self.save_model(self.writer_dir)

            # if self.curr_stage != 'pretrain':
            #     # test timer start
            #     self.test_timer.start()
            #     # test
            #     self.test_loss = self.test_iter('test')
            #     # epoch timer end
            #     self.test_timer.stop()


# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support()
#
#     import model_define
#     import preprocess
#     import torch.nn as nn
#
#     m_device_type = 'gpu'
#     m_log_dir = '.\\log'
#     m_data_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
#     m_batch_size = 64
#     m_workers = 1
#
#     m_preprocess_param = {
#         'token_tuple': (32, False, 1),
#         'rnd_para_data_path': '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle',
#         'num_classes': 8,
#     }
#     m_preprocessor = preprocess.MultiBertProcessing(**m_preprocess_param)
#
#     m_model = model_define.MyMulitBERTPreTrain
#     m_model_param = {
#         'token_tuple': (32, False, 1),
#         'rnd_token_table': '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle',
#         'batch_size': m_batch_size,
#         'embedding_token_dim': 16,
#         'max_num_seg': 5,
#         'max_token': 10000,
#         'encoder_para': (3, 4, 256),
#         'loss_fun': {
#             'mlm_loss': nn.MSELoss(),
#             'nsp_loss': nn.CrossEntropyLoss(),
#         },
#         'device': utility_function.try_gpu(),
#     }
#
#     m_optimizer_param = {
#         'optimizer_name': 'Adam',
#         'lr': 0.0001,
#         'betas': (0.9, 0.98),
#         'eps': 1e-9,
#         'weight_decay': 0,
#         'amsgrad': False,
#     }
#     m_scheduler_param = {
#         'scheduler name': 'StepLR',
#         'step_size': 10,
#         'gamma': 0.95,
#         'last_epoch': -1,
#         'verbose': False
#     }
#
#     m_trainer = Trainer(device_type=m_device_type,
#                         log_dir=m_log_dir,
#                         data_path=m_data_path,
#                         batch_size=m_batch_size,
#                         workers=m_workers,
#                         model=m_model,
#                         model_param=m_model_param,
#                         optimizer_param=m_optimizer_param,
#                         scheduler_param=m_scheduler_param)
#
#     # train_mode:('pretrain', 'train')
#     m_trainer.run(train_mode='pretrain',
#                   num_epoch=512,
#                   preprocessor=m_preprocessor)
