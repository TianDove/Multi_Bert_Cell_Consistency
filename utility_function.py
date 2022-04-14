# utility function
import os
import sys
import math
import pickle
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def round_precision(x: float, precision: int = 0) -> float:
    """精确四舍五入"""
    val = x * 10**precision
    int_part = math.modf(val)[1]
    fractional_part = math.modf(val)[0]
    out = 0
    if fractional_part >= 0.5:
        out = int_part + 1
    else:
        out = int_part
    out = out / 10**precision
    return out


def try_gpu(i: int = 0) -> torch.device:
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus() -> torch.device:
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def read_pickle_file(file_path: str) -> dict:
    """"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict


def write_to_pickle(target_path: str, data_dict: dict) -> None:
    """"""
    if not os.path.exists(target_path):
        with open(target_path, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        print(f'Path Existed: {target_path}')
        sys.exit(0)


def write_to_txt(file_path: str, data) -> None:
    """"""
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            print(data, file=file)
    else:
        with open(file_path, 'a', encoding='utf-8') as file:
            print(data, file=file)


def cell_data_processed_dynamic_plot(cell_data_set_dict: dict, save_path: str = None) -> plt.Figure:
    """"""
    first_item = cell_data_set_dict[list(cell_data_set_dict.keys())[0]]
    cell_data_dict_key = list(first_item.keys())
    if 'Static' in cell_data_dict_key:
        cell_data_dict_key.remove('Static')

    # fig init
    fig, ax = plt.subplots(5, 3, figsize=(18, 9))

    # set column title
    ax[0, 0].set_title('voltage')
    ax[0, 1].set_title('current')

    # set row title
    ax[0, 0].set_ylabel('ch1')
    ax[1, 0].set_ylabel('ch2')
    ax[2, 0].set_ylabel('ch3')
    ax[3, 0].set_ylabel('dc')
    ax[4, 0].set_ylabel('ch4')

    for cell in iter(cell_data_set_dict):
        temp_cell_data = cell_data_set_dict[cell]
        for key in iter(cell_data_dict_key):
            temp_para_data = temp_cell_data[key]
            temp_key_split = key.split('-', 1)
            temp_dyim = temp_key_split[0]
            temp_para = temp_key_split[1]
            # row set

            if temp_dyim == 'Charge #1':
                id_row = 0
            elif temp_dyim == 'Charge #2':
                id_row = 1
            elif temp_dyim == 'Charge #3':
                id_row = 2
            elif temp_dyim == 'Discharge':
                id_row = 3
            elif temp_dyim == 'Charge #4':
                id_row = 4
            else:
                raise ValueError

            # col set
            if temp_para == 'voltage':
                id_col = 0
            elif temp_para == 'current':
                id_col = 1
            else:
                raise ValueError

            temp_time = list(temp_para_data.index)
            temp_data = list(temp_para_data.values)
            with plt.ion():
                ax[id_row, id_col].plot(temp_time, temp_data, linewidth=0.5)

    if save_path is not None:
        if not os.path.exists(save_path):
            fig.savefig(f'{save_path}', format='pdf')
        else:
            raise FileExistsError
    return fig


def cell_data_selected_dynamic_plot(cell_data_set_dict: dict, save_path: str = None) -> plt.Figure:
    """"""
    assert not (cell_data_set_dict == {})

    # init plot fig
    fig, ax = plt.subplots(5, 3, figsize=(18, 9))

    # set column title
    ax[0, 0].set_title('voltage')
    ax[0, 1].set_title('current')
    ax[0, 2].set_title('capacity')

    # set row title
    ax[0, 0].set_ylabel('ch1')
    ax[1, 0].set_ylabel('ch2')
    ax[2, 0].set_ylabel('ch3')
    ax[3, 0].set_ylabel('dc')
    ax[4, 0].set_ylabel('ch4')

    # ax layout
    #       | 1 voltage | 2 current | 3 capacity |
    # | ch1 |
    # | ch2 |   green/ normal
    # | ch3 |   red  / abnormal
    # | dc  |

    # | ch4 |
    num_of_cell = len(cell_data_set_dict)
    with plt.ion():
        with tqdm(total=num_of_cell) as f_bar:
            # set f_bar description
            f_bar.set_description('Cell Selected Plotting:')
            for temp_cell in iter(cell_data_set_dict):
                temp_cell_dict = cell_data_set_dict[temp_cell]
                temp_cell_grade = temp_cell_dict['Static'].iloc[-1].at['Grade']

                # set line color
                if temp_cell_grade != 'H':
                    line_color = 'g'
                elif temp_cell_grade == 'G':
                    line_color = 'b'
                else:
                    line_color = 'r'

                for temp_key in iter(temp_cell_dict.keys()):
                    if temp_key != 'Static':
                        temp_key_df = temp_cell_dict[temp_key].iloc[1:, :].astype('float')

                        # default ax
                        ax_c = 0
                        ax_r = 0
                        lw = 0.5
                        if temp_key == 'Charge #1':
                            ax_r = 0
                        elif temp_key == 'Charge #2':
                            ax_r = 1
                        elif temp_key == 'Charge #3':
                            ax_r = 2
                        elif temp_key == 'Discharge':
                            ax_r = 3
                        elif temp_key == 'Charge #4':
                            ax_r = 4
                        else:
                            raise ValueError

                        # extract data
                        time_stamp = temp_key_df.index.tolist()
                        voltage = temp_key_df['voltage'].tolist()
                        current = temp_key_df['current'].tolist()
                        capacity = temp_key_df['capacity'].tolist()

                        # plot voltage
                        ax[ax_r, 0].plot(time_stamp, voltage, line_color, linewidth=lw)
                        # plot current
                        ax[ax_r, 1].plot(time_stamp, current, line_color, linewidth=lw)
                        # plot capacity
                        ax[ax_r, 2].plot(time_stamp, capacity, line_color, linewidth=lw)

                # update tqdm bar
                f_bar.update()
    if save_path is not None:
        if not os.path.exists(save_path):
            fig.savefig(f'{save_path}', format='pdf')
        else:
            raise FileExistsError
    return fig


def dict_slice(adict: dict, start: int, end: int) -> dict:
    """"""
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice


def plot_cluster_label(para_pad_arr: np.array, labels: np.array) -> plt.Figure:
    """"""
    # fig init
    fig, ax = plt.subplots()
    size = para_pad_arr.shape
    x_ax = [x for x in range(size[1])]
    lw = 1

    # set label color and legend
    n_counter = Counter(labels)
    labels_keys = list(n_counter.keys())

    # set color
    n_color = len(labels_keys)
    init_color = 0
    color_dict = {}
    # line_dict = {}
    for key in labels_keys:
        # color dict init
        temp_color = 'C' + str(init_color)
        color_dict.update({key: temp_color})
        init_color += 1

        # # line dict init
        # line_dict.update({key: []})
    with plt.ion():
        with tqdm(total=size[0]) as bar:
            bar.set_description('Plotting Labeled Data')
            for row in iter(range(size[0])):
                temp_label = labels[row]
                line, = ax.plot(x_ax, para_pad_arr[row, :],
                                color=color_dict[temp_label],
                                linewidth=lw,
                                label=temp_label)
                bar.update()
        # set legend
        hd, lb = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(lb, hd))
        ax.legend(by_label.values(), by_label.keys())
    return fig


def plot_maj_min_divide_line(updated_centers_df: pd.DataFrame,
                             centers: np.array,
                             n_min_c: int,
                             n_maj_c: int) -> plt.Figure:
    """"""
    fig, ax = plt.subplots()
    size = centers.shape
    x_ax = [x for x in range(size[1])]
    lw = 0.1
    with plt.ion():
        for idx in iter(list(updated_centers_df.index)):
            temp_data = np.squeeze(updated_centers_df.loc[[idx], :].values, axis=0)
            if idx == n_min_c:
                ax.plot(x_ax, temp_data, 'g', label='min border', linewidth=lw)
            if idx == n_maj_c:
                ax.plot(x_ax, temp_data, 'r', label='maj border', linewidth=lw)
            if idx == -1:
                ax.plot(x_ax, temp_data, 'b', label='divide line', linewidth=lw)
            else:
                # ax.plot(x_ax, temp_data, 'y', linewidth=lw)
                pass

        # set legend
        hd, lb = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(lb, hd))
        ax.legend(by_label.values(), by_label.keys())
        # plt.show(block=True)
        return fig


def intermedia_tensor_inspcetion(data: torch.tensor) -> np.array:
    """"""
    return data.detach().numpy()


def intermedia_data_dict_inspcetion(data_dict: dict) -> np.array:
    """"""
    # data tensor visualization
    tensor_list = []
    for key, data in iter(data_dict.items()):
        tensor_list.append(data.detach().numpy())
    vis_arr = np.concatenate(tensor_list, axis=1)
    return vis_arr


def save_model_checkpoint(model, optimizer, epoch, loss, path):
    """
    save model checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_model_checkpoint(path, model, optimizer):
    """
    load model checkpoint
    """
    checkpoint_dict = torch.load(path)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    epoch = checkpoint_dict['epoch']
    loss = checkpoint_dict['loss']
    return model, optimizer, epoch, loss


def tensor_dict_to_device(in_dict,
                          device=torch.device('cpu')):
    """"""
    out_dict = {}
    for key, value in iter(in_dict.items()):
        out_dict[key] = value.to(device)

    return out_dict


def init_model(model):
    """"""
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


def para_simple_concat(in_data: dict):
    """"""
    for key, value in iter(in_data.items()):
        if key != 'pretrain':
            temp_cell_dict = value
            temp_cat_list = []
            for para_name, para_val in iter(temp_cell_dict.items()):
                pass


def write_txt(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            print(data, file=file)
    else:
        with open(file_path, 'a', encoding='utf-8') as file:
            print(data, file=file)


def merge_dict(dict1, dict2):
    """"""
    return (dict2.update(dict1))


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
    """记录多次运行时间(秒,s)"""
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


class Tokenizer():
    """"""
    def __init__(self, token_tuple=(32, True, 16)):
        """
        token_tup = (t_len, overlap, step)
        """
        self.t_len = token_tuple[0]
        self.overlap = token_tuple[1]
        self.step = token_tuple[2]
        self.detoken_len = None
        self.num_token = None

    def tokenize(self, in_data: torch.tensor) -> torch.tensor:
        """"""
        in_temp = in_data
        d_size = in_temp.shape
        assert d_size[1] > self.t_len
        r_mod = d_size[1] % self.t_len

        if not self.overlap:
            if r_mod != 0:
                pad_num = 0
                num_of_padding = self.t_len - r_mod
                pad_arr = torch.ones(num_of_padding) * pad_num
                in_temp = torch.cat((in_temp, pad_arr.reshape(1, -1)), dim=1)
            out_data = in_temp.reshape(-1, self.t_len)
            self.num_of_token = out_data.shape[0]
            self.detoken_len = self.num_of_token * self.t_len
        else:
            num_of_step = math.ceil((d_size[0] - (self.t_len - self.step)) / self.step)
            self.detoken_len = (num_of_step - 1) * self.step + self.t_len
            if (self.detoken_len % d_size[0]) != 0:
                pad_num = 0
                num_of_padding = self.detoken_len - d_size[0]
                pad_arr = torch.ones(num_of_padding) * pad_num
                in_temp = torch.cat((in_temp, pad_arr))
            # overlap tokenize
            out_data = torch.zeros((num_of_step, self.t_len))
            for stp in range(num_of_step):
                index = stp * self.step
                temp_token = in_temp[index:index + self.t_len]
                out_data[stp, :] = temp_token
            self.num_of_token = out_data.shape[0]
        return out_data

    def detokenize(self, in_data):
        """"""
        org_size = in_data.shape
        if not self.overlap:
            out_data = in_data.view(1, -1)
        else:
            num_of_token = org_size[0]
            out_data = torch.zeros((num_of_token - 1) * self.step + self.t_len)
            first_token = in_data[0, :]
            out_data[0:self.t_len] = first_token  # put first token into out sequence
            for i in range(1, num_of_token):
                curr_token = in_data[i, :]  # get token from second token
                curr_start_index = i * self.step
                curr_end_index = curr_start_index + self.t_len
                padded_curr_token = torch.zeros((num_of_token - 1) * self.step + self.t_len)
                padded_curr_token[curr_start_index: curr_end_index] = curr_token
                out_data += padded_curr_token
                curr_mid_start_index = curr_start_index
                curr_mid_end_index = curr_start_index + self.step
                out_data[curr_mid_start_index: curr_mid_end_index] /= 2
        return out_data

    def token_wrapper(self, data, *args):
        """"""
        if args[0] == 'token':
            assert (len(data.shape) == 1) and (type(data) is np.ndarray)
            arr_token = self.tokenize(data)
        elif args[0] == 'detoken':
            # in_data is a tensor:(number of token, token length)
            assert torch.is_tensor(data) and (len(data.shape) == 2)
            arr_token = self.detokenize(data)
        else:
            raise Exception('Tokenize Mode Error.')
        # convert data
        re_data = arr_token
        return re_data


if __name__ == '__main__':
    pass


