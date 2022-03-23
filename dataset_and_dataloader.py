# std import
from datetime import datetime
import random
from collections import Counter, OrderedDict

# third party import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# app specific import
import utility_function


class DataSetLabel(object):
    """"""
    def __init__(self,
                 data_file_path: str,
                 cluster_type: str,
                 step_token: tuple):
        """"""
        self.data_file_path = data_file_path
        self.cluster_type = cluster_type
        self.cell_grade = ['A', 'B', 'C', 'D', 'E', 'F']
        self.cell_grade_list_dict = None
        self.step_token = step_token
        self.data_len_dict = {}

    def cell_in_grade_list_init(self):
        """"""
        if self.cell_grade_list_dict is None:
            self.cell_grade_list_dict = {}
            for grade in self.cell_grade:
                self.cell_grade_list_dict.update({f'{grade}': []})

    def cell_selection(self, in_dict: dict):
        """"""
        with tqdm(total=len(in_dict)) as bar:
            bar.set_description('Cell Selection')
            for cell_name in iter(in_dict):
                temp_cell_dict = in_dict[cell_name]
                temp_cell_grade = temp_cell_dict['Static']['Grade'].item()
                self.cell_grade_list_dict[temp_cell_grade].append(temp_cell_dict)
                bar.update()

        print('=' * 70)
        print(f'| Grade | Number of Samples |')
        print('=' * 70)
        for grade in self.cell_grade_list_dict:
            print(f'|   {grade}   |      {len(self.cell_grade_list_dict[grade])}      |')
        print('=' * 70)

    def data_label_grade(self, in_grade_list: list) -> list:
        """"""
        out_list = []
        first_item = in_grade_list[0]
        item_keys = in_grade_list[0].keys()

        # para_list init
        para_list_dict = {}
        for temp_key in iter(item_keys):
            if temp_key != 'Static':
                para_list_dict.update({f'{temp_key}': []})

        for cell in iter(in_grade_list):
            temp_cell_dict = cell
            for para in para_list_dict:
                temp_org_para = torch.tensor(temp_cell_dict[para].values, dtype=torch.float32)
                para_list_dict[para].append(temp_org_para)

        # para_df init
        para_df_dict = {}
        for para in para_list_dict:
            temp_para_list = para_list_dict[para]
            temp_para_pad_arr = pad_sequence(temp_para_list, True).numpy()  # B * L

            # label with cluster algorithm
            if self.cluster_type == 'DBSCAN':

                # DBSCAN setting
                epsfloat = 0.1
                min_samples = 5
                metric = 'euclidean'
                metric_params = None
                algorithm = 'kd_tree'
                leaf_sizeint = 16
                pfloat = None
                n_jobs = 1
                c_DBSCAN = DBSCAN(eps=epsfloat,
                                  min_samples=min_samples,
                                  metric=metric,
                                  metric_params=metric_params,
                                  algorithm=algorithm,
                                  leaf_size=leaf_sizeint,
                                  p=pfloat,
                                  n_jobs=n_jobs)
                clustering = c_DBSCAN.fit(temp_para_pad_arr)
                core_of_cluster = clustering.core_sample_indices_

            elif self.cluster_type == 'K-means':

                # k-means setting
                n_clusters = 6
                init = 'k-means++'
                n_init = 10
                max_iterint = 300
                tolfloat = 1e-4
                algorithm = 'full'

                # init k-means
                c_KMeans = KMeans(n_clusters=n_clusters,
                                  init=init,
                                  n_init=n_init,
                                  max_iter=max_iterint,
                                  tol=tolfloat,
                                  algorithm=algorithm)
                clustering = c_KMeans.fit(temp_para_pad_arr)
                core_of_cluster = clustering.cluster_centers_

            else:
                raise ValueError

            # # calculate divide line
            # divide_line = self.calculate_divide_line(clustering.labels_, core_of_cluster)
            # for data in temp_para_pad_arr:
            #     temp_data = data
            #     temp_dist = np.expand_dims(temp_data, axis=0) - divide_line
            #
            #     # filter
            #     sig_df = pd.DataFrame(data=temp_dist.T)
            #     sig_bool = sig_df[sig_df < 0].isnull()
            #     sig_counter = sig_bool.value_counts()
            #     if not((sig_counter[False] > 0) & (sig_counter[True] > 0)):
            #         cell_index = 1

            # fig = utility_function.plot_cluster_label(temp_para_pad_arr, clustering.labels_)
            # para_df_dict.update({f'{para}': temp_para_pad_arr})

            # concat label into data array

            temp_labels = np.expand_dims(clustering.labels_, axis=1)
            temp_out_arr = np.concatenate((temp_para_pad_arr, temp_labels), axis=1)
            temp_labeled_list = list(temp_out_arr)
            para_df_dict.update({f'{para}': temp_labeled_list})


        return out_list

    def main_data_label_all(self):
        """"""
        if os.path.exists(self.data_file_path):
            # read cell data file
            data_dict = utility_function.read_pickle_file(self.data_file_path)

            # split data into parameter array

            # init parameter array
            first_item = data_dict[list(data_dict.keys())[0]]
            item_keys = first_item.keys()

            # para_list init
            para_list_dict = {}
            for temp_key in iter(item_keys):
                if temp_key != 'Static':
                    para_list_dict.update({f'{temp_key}': []})
            with tqdm(total=len(data_dict)) as pbar:
                pbar.set_description('Data splitting')
                # implement data split
                for cell in iter(data_dict):
                    temp_cell_dict = data_dict[cell]
                    for para in para_list_dict:
                        temp_org_para = torch.tensor(temp_cell_dict[para].values, dtype=torch.float32)
                        para_list_dict[para].append(temp_org_para)

                    # update progress bar
                    pbar.update()

            # pad sequence
            para_pad_step_dict = {}
            para_pad_unstep_dict = {}
            para_list_dict_len = len(para_list_dict)
            for i, para in enumerate(para_list_dict):
                temp_para_list = para_list_dict[para]
                temp_para_pad_arr = pad_sequence(temp_para_list, True).numpy()
                para_pad_unstep_dict.update({f'{para}': temp_para_pad_arr})
                # update self.data_len_dict with adding sequence and length of each parameter
                self.data_len_dict.update({f'{para}': temp_para_pad_arr.shape[1]})

                if i != (para_list_dict_len - 1):
                    temp_arr_shape = temp_para_pad_arr.shape
                    temp_step_arr = np.ones([temp_arr_shape[0], self.step_token[1]]) * self.step_token[0]
                    temp_para_pad_arr = np.concatenate((temp_para_pad_arr, temp_step_arr), axis=1)
                para_pad_step_dict.update({f'{para}': temp_para_pad_arr})

            # concat element of dict into one array
            concat_para_pad_step_arr = np.concatenate(list(para_pad_step_dict.values()), axis=1)

            # label with cluster algorithm
            if self.cluster_type == 'DBSCAN':

                # DBSCAN setting
                epsfloat = 0.1
                min_samples = 5
                metric = 'euclidean'
                metric_params = None
                algorithm = 'kd_tree'
                leaf_sizeint = 16
                pfloat = None
                n_jobs = 1
                c_DBSCAN = DBSCAN(eps=epsfloat,
                                  min_samples=min_samples,
                                  metric=metric,
                                  metric_params=metric_params,
                                  algorithm=algorithm,
                                  leaf_size=leaf_sizeint,
                                  p=pfloat,
                                  n_jobs=n_jobs)
                clustering = c_DBSCAN.fit(concat_para_pad_step_arr)
                core_of_cluster = clustering.core_sample_indices_

            elif self.cluster_type == 'K-means':

                # k-means setting
                n_clusters = 8
                init = 'k-means++'
                n_init = 10
                max_iterint = 300
                tolfloat = 1e-4
                algorithm = 'full'

                # init k-means
                c_KMeans = KMeans(n_clusters=n_clusters,
                                  init=init,
                                  n_init=n_init,
                                  max_iter=max_iterint,
                                  tol=tolfloat,
                                  algorithm=algorithm)
                clustering = c_KMeans.fit(concat_para_pad_step_arr)
                core_of_cluster = clustering.cluster_centers_
            else:
                raise ValueError

            # concat label and data
            para_pad_unstep_dict.update({'Labels': clustering.labels_})

            data_set_index_dict = self.split_dataset(para_pad_unstep_dict)

            para_pad_unstep_dict.update({'DataSetIndex': data_set_index_dict})
            # save data
            utility_function.write_to_pickle('.\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels.pickle',
                                             para_pad_unstep_dict)


        else:
            raise FileNotFoundError

    def split_dataset(self,
                      pad_step_labels_arr,
                      pretrain=0.65,
                      supervise=0.35,
                      train=0.8,
                      val=0.1,
                      test=0.1) -> dict:
        """"""

        if type(pad_step_labels_arr) is not dict:
            num_samples = pad_step_labels_arr.shape[0]
        else:
            num_samples = pad_step_labels_arr[list(pad_step_labels_arr.keys())[0]].shape[0]

        num_pretrain = int(num_samples * pretrain)
        num_supervise = num_samples - num_pretrain

        num_train = int(num_supervise * train)
        num_val = int(num_supervise * val)
        num_test = num_supervise - num_train - num_val

        # shuffle data index
        data_index_list = list(range(num_samples))
        random.shuffle(data_index_list)


        # split data
        pretrain_data_index = data_index_list[:num_pretrain]
        supervise_data_index = data_index_list[num_pretrain:]

        train_data_index = supervise_data_index[:num_train]
        val_data_index = supervise_data_index[num_train:num_train + num_val]
        test_data_index = supervise_data_index[num_train + num_val:]

        temp_data_set_dict = {
            'pretrain': pretrain_data_index,
            'train': train_data_index,
            'val': val_data_index,
            'test': test_data_index
            }

        return temp_data_set_dict

    def main_data_label(self):
        """"""
        if os.path.exists(self.data_file_path):
            # self.cell_in_grade_list_init()
            data_dict = utility_function.read_pickle_file(self.data_file_path)
            # self.cell_selection(data_dict)

            # test only
            # temp_file_name = '.\\pik\\test_2022-03-05-12-35-45_Cell_set_LayerStd-cell_grade_list_dict.pickle'
            # self.cell_grade_list_dict = utility_function.read_pickle_file(temp_file_name)
            #

            grade_labeled_list_dict = {}
            for grade in iter(self.cell_grade_list_dict):
                temp_grade_list = self.cell_grade_list_dict[grade]
                if temp_grade_list:
                    temp_grade_labeled_list = self.data_label_grade(temp_grade_list)
                    grade_labeled_list_dict.update({f'{grade}': temp_grade_labeled_list})
        else:
            raise FileExistsError

    def calculate_divide_line(self, labels: np.array, centers: np.array) -> np.array:
        """"""
        count_res = Counter(labels)
        centers_df = pd.DataFrame(centers, index=[x for x in range(centers.shape[0])])

        # dict sort by values
        sorted_res = {k: v for k, v in sorted(count_res.items(), key=lambda item: item[1], reverse=True)}

        mid_index = len(sorted_res.keys()) // 2
        dict_f = utility_function.dict_slice(sorted_res, 0, mid_index)
        dict_b = utility_function.dict_slice(sorted_res, mid_index, len(sorted_res.keys()))

        # spilt center into maj and min
        maj_center_list = []
        min_center_list = []
        for key in dict_f.keys():
            maj_center_list.append(centers_df.loc[[key], :])
        for key in dict_b.keys():
            min_center_list.append(centers_df.loc[[key], :])

        maj_center_df = pd.concat(maj_center_list, axis=0)
        min_center_df = pd.concat(min_center_list, axis=0)

        # min - row
        # maj - col
        dist_df = pd.DataFrame(index=min_center_df.index, columns=maj_center_df.index)
        for maj_c_idx in list(maj_center_df.index):
            temp_maj_c = maj_center_df.loc[[maj_c_idx], :]
            for min_c_idx in list(min_center_df.index):
                temp_min_c = min_center_df.loc[[min_c_idx], :]
                dist = euclidean_distances(temp_maj_c.values, temp_min_c.values).item(0)
                dist_df.loc[min_c_idx, maj_c_idx] = dist
                
        dist_np = dist_df.astype('float').to_numpy()
        min_dist_index = np.where(dist_np == np.min(dist_np))

        n_maj_c = list(maj_center_df.index)[min_dist_index[1][0]]
        n_min_c = list(min_center_df.index)[min_dist_index[0][0]]

        border_maj_center = centers_df.loc[[n_maj_c], :]
        border_min_center = centers_df.loc[[n_min_c], :]
        divide_center = (border_maj_center.values + border_min_center.values) / 2

        # plot maj center, min center and divide_center
        divide_center_df = pd.DataFrame(data=divide_center, index=[-1])
        updated_centers_df = centers_df.append(divide_center_df)

        # plot maj, min and divide line
        fig = utility_function.plot_maj_min_divide_line(updated_centers_df,
                                                        centers,
                                                        n_min_c,
                                                        n_maj_c)

        # 1 * L
        return divide_center

    def plot_pad_step_labels_arr(self, filename: str, pad_step_labels_arr: np.array) -> None:
        """"""
        # figure and axes initialization
        fig, ax = plt.subplots()
        x_axis = np.arange(pad_step_labels_arr.shape[1])

        # plot pad step labels
        with plt.ion():
            with tqdm(total=pad_step_labels_arr.shape[0]) as bar:
                bar.set_description('Plotting Labeled Data')
                for row in iter(pad_step_labels_arr):
                    ax.plot(x_axis, row, linewidth=0.5)

                    # update bar
                    bar.update()
                fig.savefig(f'.\\{filename}.pdf')

    def data_set_resampling(self,
                            data_arr: np.array,
                            resampling_method: str,
                            resample_rate: [float]) -> np.array:
        """"""
        pass

    def form_data_set_cell_dict(self, un_formed_dict: dict) -> dict:
        """"""
        formed_data_set_cell_dict = {
            'pretrain': {},
            'train': {},
            'val': {},
            'test': {}
        }

        for data_set in formed_data_set_cell_dict:
            temp_data_set_index = un_formed_dict['DataSetIndex'][data_set]

            for cell in iter(temp_data_set_index):
                temp_cell_data_dict = {}

                temp_ch1_data = un_formed_dict['Charge #1-voltage'][cell, :]
                temp_ch2_data = un_formed_dict['Charge #2-voltage'][cell, :]
                temp_dc_data = un_formed_dict['Discharge-voltage'][cell, :]
                temp_ch31_data = un_formed_dict['Charge #3-voltage'][cell, :]
                temp_ch32_data = un_formed_dict['Charge #3-current'][cell, :]
                temp_label = un_formed_dict['Labels'][cell]

                temp_cell_data_dict.update({'ch1v': temp_ch1_data})
                temp_cell_data_dict.update({'ch2v': temp_ch2_data})
                temp_cell_data_dict.update({'dcv': temp_dc_data})
                temp_cell_data_dict.update({'ch3v': temp_ch31_data})
                temp_cell_data_dict.update({'ch3c': temp_ch32_data})
                temp_cell_data_dict.update({'label': temp_label})

                formed_data_set_cell_dict[data_set].update({cell: temp_cell_data_dict})

        return formed_data_set_cell_dict


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
                          is_shuffle: bool,
                          num_of_worker: int) -> DataLoader:
        """"""
        data_set = cls(data_dict, type_data_set)
        data_loader = DataLoader(data_set,
                                 batch_size=batch_sz,
                                 shuffle=is_shuffle,
                                 num_workers=num_of_worker,
                                 pin_memory=True)
        return data_loader


if __name__ == '__main__':
    import os
    import sys

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.environ['OMP_NUM_THREADS'] = '1'

    data_file_base = '.\\pik'
    curr_file_name = '2022-03-05-13-36-24_Cell_set_MinMax.pickle'
    m_data_file_path = os.path.join(data_file_base, curr_file_name)

    # Cluster type
    # ('DBSCAN', 'K-means')
    m_cluster_type = 'K-means'

    m_temp_step_token = (-1, 16)

    # init class
    m_data_label = DataSetLabel(m_data_file_path,
                                m_cluster_type,
                                m_temp_step_token)

    # run
    # m_data_label.main_data_label_all()
    m_data_file_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_data_dict = utility_function.read_pickle_file(m_data_file_path)

    # temp_data_dict = {}
    # for key in m_data_dict:
    #     temp_dataset = m_data_dict[key]
    #     temp_slice_dict = utility_function.dict_slice(temp_dataset, 0, 100)
    #     temp_data_dict.update({key: temp_slice_dict})

    # formed_dict = m_data_label.form_data_set_cell_dict(m_data_dict)
    # ('pretrain', 'train', 'val', 'test')
    m_data_loader = CellDataLoader.creat_data_loader(m_data_dict, 'pretrain', 32, True, 1)

    for i, data in enumerate(m_data_loader):
        temp_data = data
    sys.exit(0)