#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import math
import time
import os

import optuna
import torch
import numpy as np
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as skl_mtr

import file_operation
import functional


class ClsRun():
    """"""
    def __init__(self,
                 data_time: str,
                 data_set_dic: dict,
                 n_epochs: int,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 writer: SummaryWriter,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler = None,
                 device: torch.device = torch.device('cpu'),
                 hyper_para_dict: dict = None,
                 trial: optuna.trial.Trial = None):
        """"""
        # save input
        self.data_time = data_time
        self.data_set_dic = data_set_dic
        self.n_epochs = n_epochs
        self.model = model
        self.loss_fn = loss_fn
        self.writer = writer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.hyper_para_dict = hyper_para_dict
        self.trial = trial
        self.test_bsz = None

        # get data set
        self.train_data_set = self.data_set_dic['train_data_set']
        self.n_train_batch = len(self.train_data_set)
        self.val_data_set = self.data_set_dic['val_data_set']
        self.n_val_batch = len(self.val_data_set)
        self.test_data_set = self.data_set_dic['test_data_set']
        self.n_test_batch = len(self.test_data_set)

        # save dir
        self.save_dir = self.writer.log_dir
        # get model name
        self.model_name = model.model_name

        # record out and label
        self.batch_train_list = []
        self.val_dict = []
        self.test_dict = []

        # record loss
        self.batch_train_loss = 0.0
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.test_loss = 0.0
        self.epoch_val_accu_list = []

        # record accuracy
        self.epoch_train_accu = 0.0
        self.epoch_val_accu = 0.0
        self.test_accu = 0.0

        # record time
        self.epoch_start_time = 0.0
        self.batch_train_start_time = 0.0
        self.batch_train_end_time = 0.0
        self.epoch_train_end_time = 0.0
        self.epoch_val_start_time = 0.0
        self.epoch_end_time = 0.0
        self.test_start_time = 0.0
        self.test_end_time = 0.0

        self.model_start_time = 0.0
        self.model_end_time = 0.0
        self.model_time_list = []

        # current loop state
        self.curr_epoch = None
        self.curr_batch = None
        self.curr_stage = None
        self.writer_graph_flag = False
        self.writer_hypara_flag = False
        self.total_steps = 0

    def run(self):
        """"""
        for epoch in range(self.n_epochs):
            self.epoch_start_time = time.perf_counter()
            self.curr_epoch = epoch

            # training
            self.epoch_train_loss = 0.0
            self.epoch_train_loss, self.epoch_train_accu = self.train_loop()
            self.epoch_train_end_time = time.perf_counter()

            # validation
            self.epoch_val_loss = 0.0
            self.epoch_val_start_time = time.perf_counter()
            self.epoch_val_loss, self.epoch_val_accu, self.val_dict = self.test_loop('val',
                                                                                     self.val_data_set,
                                                                                     self.n_val_batch)
            self.epoch_val_accu_list.append(self.epoch_val_accu)
            self.epoch_end_time = time.perf_counter()

            if self.scheduler is not None:
                self.scheduler.step()

            self.run_log(self.save_dir, 'val', is_print=True)  # log val

            # write metric to tensorboard
            self.write_eval(self.val_dict)

            # save model
            self.save_model(self.save_dir)

            # report trial
            if self.trial is not None:
                self.trial.report(self.epoch_val_accu, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        self.write_hypara()

    def test_run(self, epoch, test_bsz):
        """"""
        self.curr_epoch = epoch
        self.test_bsz = test_bsz

        self.test_loss = 0.0
        self.test_start_time = time.perf_counter()
        self.test_loss, self.test_accu, self.test_dict = self.test_loop('test', self.test_data_set, self.n_test_batch)
        self.test_end_time = time.perf_counter()
        self.run_log(self.save_dir, 'test')  # log test"""
        # write metric to tensorboard
        self.test_write_eval(self.test_dict)

    def write_hypara(self):
        """"""
        # writer current hyper-parameter
        if not self.writer_hypara_flag:
            self.writer_hypara_flag = True
            if self.trial is not None:
                self.writer.add_hparams(self.hyper_para_dict,
                                        {'Max Val Accu': max(self.epoch_val_accu_list)},
                                        run_name=str(self.trial.number))
            else:
                self.writer.add_hparams(self.hyper_para_dict,
                                        {'Max Val Accu': max(self.epoch_val_accu_list)})

    def train_loop(self):
        """"""
        self.curr_stage = 'train'
        self.model.train()
        sum_loss = 0.0
        out_list = []
        label_list = []
        for batch_index, train_data in enumerate(self.train_data_set):
            self.curr_batch = batch_index
            self.batch_train_start_time = time.perf_counter()
            data, label = train_data
            data = data.to(device=self.device)
            label = label.to(device=self.device)
            if not self.writer_graph_flag:
                self.writer.add_graph(self.model, data)
                self.writer_graph_flag = True
            out = self.model(data)
            scalar_loss = self.loss_fn(out, label)
            self.optimizer.zero_grad()
            scalar_loss.backward()
            self.optimizer.step()
            # record loss
            sum_loss += scalar_loss.item()
            out_list.append(torch.argmax(out, dim=1))
            label_list.append(label)
            # write batch loss
            # self.writer.add_scalars(f'{self.model_name}_Batch_Scalar',
            #                        {'train_batch_loss': scalar_loss.item()},
            #                        self.total_steps)
            self.total_steps += 1
            self.batch_train_loss = scalar_loss.item()
            self.batch_train_end_time = time.perf_counter()
            self.run_log(self.save_dir, self.curr_stage, is_print=True)  # log train
        # calculate train avg accu
        accu = 0.0
        total_out = torch.cat(out_list).to(device=torch.device('cpu')).numpy()
        total_label = torch.cat(label_list).to(device=torch.device('cpu')).numpy()
        accu = skl_mtr.accuracy_score(total_label, total_out)

        return sum_loss / self.n_train_batch, accu

    def test_loop(self, stage, data_set, n_batch):
        """"""
        self.model.eval()
        self.curr_stage = stage
        self.model_time_list = []
        sum_loss = 0
        out_list = []
        label_list = []
        raw_out_list = []
        with torch.no_grad():
            for idx, zip_data in enumerate(data_set):
                data, label = zip_data
                data = data.to(device=self.device)
                label = label.to(device=self.device)

                self.model_start_time = time.perf_counter()
                out = self.model(data)
                self.model_end_time = time.perf_counter()
                self.model_time_list.append((self.model_end_time - self.model_start_time)/self.test_bsz)

                scalar_loss = self.loss_fn(out, label)
                # record loss
                sum_loss += scalar_loss.item()
                raw_out_list.append(out)
                out_list.append(torch.argmax(out, dim=1))
                label_list.append(label)
            # calculate train avg accu
            accu = 0.0
            total_out = torch.cat(out_list).to(device=torch.device('cpu')).numpy()
            total_label = torch.cat(label_list).to(device=torch.device('cpu')).numpy()
            accu = skl_mtr.accuracy_score(total_label, total_out)
            raw_out_dict = {
                'raw_out': raw_out_list,
                'label': label_list
            }
        return sum_loss / n_batch, accu, raw_out_dict

    def run_log(self, path, mode: str = 'train', is_print: bool = True):
        """"""
        str_data = ''
        separator = ''
        n_sp = 128
        if mode == 'train':
            # log batch train
            separator = '-' * n_sp
            str_data = '| Epoch {:3d}/{:3d} | Batch: {:5d}/{:5d} | lr {:10.9f} | {:5.2f} ms/batch  | ' \
                       'Train Loss {:8.7f} |'.format(
                        self.curr_epoch, self.n_epochs,
                        self.curr_batch, self.n_train_batch,
                        self.optimizer.param_groups[0]['lr'],
                        (self.batch_train_end_time - self.batch_train_start_time) * 1000,
                        self.batch_train_loss)

        elif mode == 'val':
            # log val
            separator = '-' * n_sp
            str_data = '| End of epoch {:3d} | Current Learning Rat: {:8.7f} | \n' \
                       '| Train Total Time: {:5.2f} s | Train Avg Loss: {:8.7f} | Train Accuracy {:5.4f} | \n' \
                       '| Val Total Time: {:5.2f} s | Valid Avg Loss: {:8.7f} | Valid Accuracy {:5.4f}  |'.format(
                        self.curr_epoch, self.optimizer.param_groups[0]['lr'],
                        self.epoch_train_end_time - self.epoch_start_time,
                        self.epoch_train_loss, self.epoch_train_accu,
                        self.epoch_end_time - self.epoch_val_start_time,
                        self.epoch_val_loss, self.epoch_val_accu)

        elif mode == 'test':
            # log test
            separator = '#' * n_sp
            str_data = '| Test Stage | Test Total Time: {:5.2f} s | ' \
                       'Test Avg Loss: {:8.7f} |Test Accuracy {:5.4f} | '.format(
                        self.test_end_time - self.test_start_time,
                        self.test_loss,
                        self.test_accu)
        else:
            raise ValueError('Input Log Mode Error.')

        # write to file
        if str_data != '':
            if self.trial is None:
                text_file_name = f'{self.model_name}_logging.txt'
            else:
                text_file_name = f'{self.trial.number}_{self.model_name}_logging.txt'
            log_file_path = os.path.join(path, text_file_name)
            file_operation.write_txt(log_file_path, separator)
            file_operation.write_txt(log_file_path, str_data)
            file_operation.write_txt(log_file_path, separator)
            # print
            if is_print:
                print(separator)
                print(str_data)
                print(separator)
        else:
            raise ValueError('String Data Container Empty Error.')

    def write_eval(self, raw_out_dict: dict):
        """"""

        raw_out_arr = torch.cat(raw_out_dict['raw_out']).to(device=torch.device('cpu'))
        label_arr = torch.cat(raw_out_dict['label']).to(device=torch.device('cpu')).numpy()
        score_arr = torch.argmax(raw_out_arr, dim=1)

        auc = skl_mtr.roc_auc_score(label_arr, score_arr)
        # write to tensorboard
        if self.trial is not None:
            scalar_dic = {
                f'{self.trial.number}_train_loss': self.epoch_train_loss,
                f'{self.trial.number}_val_loss': self.epoch_val_loss,
                f'{self.trial.number}_train_acc': self.epoch_train_accu,
                f'{self.trial.number}_val_acc': self.epoch_val_accu,
                f'{self.trial.number}_AUC': auc
            }
        else:
            scalar_dic = {
                'train_loss': self.epoch_train_loss,
                'val_loss': self.epoch_val_loss,
                'train_acc': self.epoch_train_accu,
                'val_acc': self.epoch_val_accu,
                'AUC': auc
            }

        self.writer.add_scalars(
            f'{self.model_name}_Scalar',
            scalar_dic,
            self.curr_epoch
        )

    def test_write_eval(self, raw_out_dict: dict):
        """"""
        raw_out_cpu = torch.cat(raw_out_dict['raw_out']).to(device=torch.device('cpu'))
        raw_out_arr = raw_out_cpu.numpy()
        label_arr = torch.cat(raw_out_dict['label']).to(device=torch.device('cpu')).numpy()
        index_arr = torch.argmax(raw_out_cpu, dim=1, keepdim=True)
        out_prob_arr = torch.gather(raw_out_cpu, 1, index_arr).numpy()
        out_arr = index_arr.squeeze().numpy()

        num_test = len(self.model_time_list)
        avg_time = sum(self.model_time_list) / num_test

        cfm = skl_mtr.confusion_matrix(label_arr, out_arr, labels=[0, 1])
        tn, fp, fn, tp = skl_mtr.confusion_matrix(label_arr, out_arr).ravel()
        g_means = math.sqrt((tp/(tp+fn))*(tn/(tn+fp)))
        mr_neg, mr_total = functional.miss_rate(label_arr, out_arr)
        pre = skl_mtr.precision_score(label_arr, out_arr, pos_label=1)
        rec = skl_mtr.recall_score(label_arr, out_arr, pos_label=1)
        precision, recall, pr_thr = skl_mtr.precision_recall_curve(label_arr, raw_out_arr[:, 1], pos_label=1)
        fpr, tpr, threshold = skl_mtr.roc_curve(label_arr, raw_out_arr[:, 1], pos_label=1)
        auc = skl_mtr.roc_auc_score(label_arr, raw_out_arr[:, 1])
        f1 = skl_mtr.f1_score(label_arr, out_arr, pos_label=1)

        # save eval dict
        eval_dict = {
            'Acc': self.test_accu,
            'Miss Rate- neg': mr_neg,
            'Miss Rate- total': mr_total,
            'Precision': pre,
            'Recall': rec,
            'PR_precision': precision,
            'PR_recall': recall,
            'PR_threshold': pr_thr,
            'ROC_fpr': fpr,
            'ROC_tpr': tpr,
            'ROC_threshold': threshold,
            'ROC_AUC': auc,
            'F1_score': f1,
            'G-Means': g_means,
            'Avg_run_time': avg_time
        }
        print(f'Model: {self.model_name}, Avg Run Time: {avg_time * 1000} ms')
        file_name = f'{self.model_name}_{self.data_time}_{self.curr_epoch}_eval_dict'
        data_type = '.pt'
        save_path = os.path.join(self.save_dir, file_name + data_type)
        torch.save(eval_dict, save_path)

        # write to tensorboard
        scalar_dic = {
            'Acc': self.test_accu,
            'Miss Rate- neg': mr_neg,
            'Miss Rate- total': mr_total,
            'Precision': pre,
            'Recall': rec,
            'ROC_AUC': auc,
            'F1_score': f1,
            'G-Means': g_means
        }
        self.writer.add_scalars(
            f'Scalars',
            scalar_dic,
            self.curr_epoch
        )

    def save_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        file_name = f'{self.model_name}_{self.data_time}_{self.curr_epoch}'
        if self.trial is not None:
            file_name = f'{self.trial.number}_{self.model_name}_{self.data_time}_{self.curr_epoch}'
        data_type = '.pt'
        save_path = os.path.join(path, file_name + data_type)
        torch.save({
            'model_name': self.model_name,
            'epoch': self.curr_epoch,
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'train_loss': self.epoch_train_loss,
            'val_loss': self.epoch_val_loss,
            'train_accu': self.epoch_train_accu,
            'vall_accu': self.epoch_val_accu,
            **self.hyper_para_dict
        }, save_path)

    def load_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        temp_dict = torch.load(path)
        self.model_name = temp_dict['model_name']
        self.model = temp_dict['model']
        self.optimizer = temp_dict['optimizer']
        self.scheduler = temp_dict['scheduler']

