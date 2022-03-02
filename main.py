#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import utility_function


def fib(n: int, w: int = 0):
  pass


if __name__ == '__main__':
    data_file_base = 'D:\\workspace\\battery_dataset\\2600P-01_DataSet\\organized_data\\'
    curr_file_type = 'pickle'
    data_file_path = os.path.join(data_file_base, curr_file_type)
    cell_file_list = os.listdir(data_file_path)

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
    with plt.ion():
        with tqdm(total=len(cell_file_list)) as f_bar:
            # set f_bar description
            f_bar.set_description('Cell Data Plotting:')
            for temp_cell_file in iter(cell_file_list):
                temp_file_path = os.path.join(data_file_path, temp_cell_file)
                temp_cell_data_dict = utility_function.read_pickle_file(temp_file_path)
                temp_cell_grade = temp_cell_data_dict['Static'].iloc[-1].at['Grade']

                # set line color
                if temp_cell_grade != 'H':
                    line_color = 'g'
                else:
                    line_color = 'r'

                for temp_key in iter(temp_cell_data_dict.keys()):
                    if temp_key != 'Static':
                        temp_key_df = temp_cell_data_dict[temp_key].iloc[1:, :].astype('float')
                        duplicated_droped_df = temp_key_df.drop_duplicates(subset='time', keep='last')
                        aligned_df = duplicated_droped_df[duplicated_droped_df['time'] % 0.5 == 0.0]
                        interpolated_df = aligned_df.interpolate(method='linear', axis=0)

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
                        time_stamp = interpolated_df['time'].tolist()
                        voltage = interpolated_df['voltage'].tolist()
                        current = interpolated_df['current'].tolist()
                        capacity = interpolated_df['capacity'].tolist()

                        # plot voltage
                        ax[ax_r, 0].plot(time_stamp, voltage, line_color, linewidth=lw)
                        # plot current
                        ax[ax_r, 1].plot(time_stamp, current, line_color, linewidth=lw)
                        # plot capacity
                        ax[ax_r, 2].plot(time_stamp, capacity, line_color, linewidth=lw)

                # update tqdm bar
                f_bar.update()
        # plt.savefig('./data_set.svg', format='svg')
        # plt.savefig('./data_set.pdf', format='pdf')
    # end run
    sys.exit(0)

