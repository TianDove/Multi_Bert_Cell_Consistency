# std import
import os
import sys
import math

# third party import
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# app specific import
import utility_function


# main object
class DataPreProcess(object):
    """"""

    def __init__(self,
                 file_path: str,
                 data_type: tuple,
                 std_type: str = None,
                 resampling_type: str = None) -> None:
        self.data_file_path = file_path
        self.data_type = data_type
        self.std_type = std_type
        self.resampling_type = resampling_type
        self.current_cell_name = None
        self.refined_cell_data_dict = None

    def data_process(self) -> None:
        """"""
        if os.path.exists(self.data_file_path):
            file_list = os.listdir(self.data_file_path)
            with tqdm(total=len(file_list)) as f_bar:
                # set f_bar description
                f_bar.set_description('Cell Data Pre-Processing:')
                # G_num = 166
                # H_num = 2088

                # refined_cell_data_dict init
                self.refined_cell_data_dict = {}
                for temp_data_type in self.data_type:
                    if temp_data_type in ['Charge #1', 'Charge #2', 'Discharge']:
                        self.refined_cell_data_dict.update({f'{temp_data_type}-voltage': []})
                    elif temp_data_type in ['Charge #3', 'Charge #4']:
                        self.refined_cell_data_dict.update({f'{temp_data_type}-voltage': []})
                        self.refined_cell_data_dict.update({f'{temp_data_type}-current': []})
                    else:
                        raise ValueError

                # cell loop
                cell_set = {}
                for temp_file_name in iter(file_list):
                    self.current_cell_name = os.path.splitext(temp_file_name)[0]
                    temp_file_path = os.path.join(self.data_file_path, temp_file_name)
                    temp_cell_data_dict = utility_function.read_pickle_file(temp_file_path)
                    temp_cell_grade = temp_cell_data_dict['Static'].iloc[-1].at['Grade']
                    if not (temp_cell_grade in ['G', 'H']):
                        ver_cell_dict = self.cell_data_verification(temp_cell_data_dict)
                        if ver_cell_dict != {}:
                            # update cell_set
                            cell_set.update({self.current_cell_name: ver_cell_dict})

                            # | Type     | used Parameter |
                            # | Charge 1 | voltage        |
                            # | Charge 2 | voltage        |
                            # | Charge 3 | voltage & current |
                            # | Discharge| voltage |
                            # | Charge 4 | voltage & current |

                            # append data
                            self.cell_data_append(ver_cell_dict, self.refined_cell_data_dict, norm_type='LayerStd')

                    # update tqdm bar
                    f_bar.update()
                # utility_function.write_to_pickle('.\\pik\\cell_set.pickle', cell_set)
                utility_function.write_to_pickle('.\\pik\\cell_data_list_layernoraml.pickle',
                                                 self.refined_cell_data_dict)
                # cell_set_fig = utility_function.cell_data_data_dynamic_plot(cell_set,
                #                                                             '.\\images\\verified_data_set.pdf')

           # cell_data_df_dict init
           #  cell_data_df_dict = {}
           #  for temp_data_type in self.data_type:
           #      if temp_data_type in ['Charge #1', 'Charge #2', 'Discharge']:
           #          cell_data_df_dict.update({f'{temp_data_type}-voltage': None})
           #      elif temp_data_type in ['Charge #3', 'Charge #4']:
           #          cell_data_df_dict.update({f'{temp_data_type}-voltage': None})
           #          cell_data_df_dict.update({f'{temp_data_type}-current': None})
           #      else:
           #          raise ValueError
           #
           #  # process loop
           #  for temp_data_type in iter(self.refined_cell_data_dict):
           #      temp_data_type_dict = self.refined_cell_data_dict[temp_data_type]
           #      temp_data_type_list = temp_data_type_dict
           #      temp_data_type_df = pd.concat(temp_data_type_list, axis=1)
           #      # temp_data_type_df = temp_data_type_df.interpolate(method='linear', axis=0)
           #      # add to dict
           #      cell_data_df_dict[temp_data_type] = temp_data_type_df



    def cell_data_verification(self, cell_data_dict) -> dict:
        """"""
        cell_data_dict_keys = cell_data_dict.keys()
        ver_cell_data_dict = {}
        ver_cell_data_dict.update({'Static': cell_data_dict['Static']})
        absent_flag = False
        ver_flag = False
        txt_seperator = '-' * 45
        for temp_data_type in self.data_type:
            if temp_data_type not in cell_data_dict_keys:
                absent_flag = True
                # absent err write
                utility_function.write_to_txt('.//ver_err_log.txt', txt_seperator)
                absent_err_str = f'{self.current_cell_name}, {temp_data_type} data missing.'
                utility_function.write_to_txt('.//ver_err_log.txt',  absent_err_str)
                utility_function.write_to_txt('.//ver_err_log.txt', txt_seperator)
                break
        if not absent_flag:
            for temp_data_type in self.data_type:
                # ver_df = None
                if temp_data_type != 'Static':
                    temp_cell_data_df = cell_data_dict[temp_data_type].iloc[1:, :].astype('float')
                    # extract static data
                    end_capacity = int(cell_data_dict['Static'][temp_data_type].iloc[-1])
                    end_current = int(cell_data_dict['Static'][temp_data_type + '.1'].iloc[-1])
                    end_voltage = int(cell_data_dict['Static'][temp_data_type + '.2'].iloc[-1])
                    # dynamic_end_index = None
                    # end_row_dynamic = None
                    if temp_data_type == 'Charge #1':  # | time | voltage | current | capacity |
                        # 'Charge #1': Constant Current Charge, 180min, 0.15C, cut-off at 3.7V
                        # extract dynamic data
                        end_row_dynamic = temp_cell_data_df.iloc[-1, :]
                        dynamic_end_index = -1

                    elif temp_data_type in ['Charge #2', 'Charge #3', 'Discharge', 'Charge #4']:
                        # 'Charge #2': Constant Current Charge, 100min, 0.5C, cut-off at 4.0V
                        # 'Charge #3': Constant Current Voltage Charge, 150min, 0.2C, cut-off at 4.2V or 0.02C
                        # 'Discharge': Constant Current Discharge, 150min, 0.5C, cut-off at 2.75V
                        # 'Charge #4': Constant Current Voltage Charge, 60min, 0.5C, cut-off at 3.55V or 0.02C
                        # extract dynamic data

                        # dynamic_current_head_10 = temp_cell_data_df['current'].iloc[0:10]
                        # extract_zero = dynamic_current_head_10.loc[(dynamic_current_head_10 == 0.0)]

                        temp_cell_data_df_ex_11 = temp_cell_data_df.iloc[11:, :]
                        temp_cell_data_df_ex_11_current = temp_cell_data_df_ex_11['current']
                        extract_zero = temp_cell_data_df_ex_11_current.loc[(temp_cell_data_df_ex_11_current == 0.0)]
                        if not extract_zero.empty:
                            dynamic_end_index = extract_zero.index.to_list()[0] - 1
                            end_row_dynamic = temp_cell_data_df.loc[dynamic_end_index, :]
                        else:
                            dynamic_end_index = -1
                            end_row_dynamic = temp_cell_data_df.iloc[-1, :]

                    else:
                        print(f'Current Cell Data Type:{temp_data_type} Not Exist.')
                        sys.exit(0)

                    dynamic_capacity = int(utility_function.round_precision(end_row_dynamic['capacity']))
                    dynamic_current = int(utility_function.round_precision(end_row_dynamic['current']))
                    dynamic_voltage = int(utility_function.round_precision(end_row_dynamic['voltage'] * 1000))
                    ver_flag = False
                    error_range = 3.0
                    if (abs(end_capacity - dynamic_capacity) <= error_range) \
                            & (abs(end_current - dynamic_current) <= error_range) \
                            & (abs(end_voltage - dynamic_voltage) <= error_range):
                        ver_flag = True
                        ver_df = temp_cell_data_df
                        if dynamic_end_index is not None:
                            processed_df = DataPreProcess.drop_duplicated_align_interpolation_data(ver_df,
                                                                                                   dynamic_end_index)
                            # update ver_dict
                            ver_cell_data_dict.update({temp_data_type: processed_df})
                    else:
                        ver_flag = False
                        # write err log
                        utility_function.write_to_txt('.//ver_err_log.txt', txt_seperator)
                        ver_err_str = self.current_cell_name + '_' + temp_data_type + '_' + 'Verification Failed.'
                        utility_function.write_to_txt('.//ver_err_log.txt', ver_err_str)
                        utility_function.write_to_txt('.//ver_err_log.txt', txt_seperator)
                        ver_err_str = f'          | voltage | current | capacity |'
                        utility_function.write_to_txt('.//ver_err_log.txt', ver_err_str)
                        ver_err_str = f'| Static  | {end_voltage} | {end_current} | {end_capacity} |'
                        utility_function.write_to_txt('.//ver_err_log.txt', ver_err_str)
                        ver_err_str = f'| Dynamic | {dynamic_voltage} | {dynamic_current} | {dynamic_capacity} |'
                        utility_function.write_to_txt('.//ver_err_log.txt', ver_err_str)
                        utility_function.write_to_txt('.//ver_err_log.txt', txt_seperator)
                        break

        if ver_flag:
            return ver_cell_data_dict
        else:
            return {}

    @staticmethod
    def drop_duplicated_align_interpolation_data(data_df: pd.DataFrame,
                                                 end_index: int) -> pd.DataFrame:
        """"""
        if end_index == -1:
            cut_end_data_df = data_df.iloc[0:, :]
        else:
            cut_end_data_df = data_df.iloc[0:end_index, :]

        # drop same date
        duplicated_droped_df = cut_end_data_df.drop_duplicates(subset='time', keep='last')

        # calculate ref time step
        temp_max_time_step = duplicated_droped_df['time'].max()
        if temp_max_time_step % 0.5 == 0:
            temp_max_ref_time_step = temp_max_time_step
        else:
            temp_max_ref_time_step = math.ceil(temp_max_time_step)

        temp_ref_time_step_list_len = int(temp_max_ref_time_step // 0.5) + 1
        temp_ref_time_step_list = []
        init_step = 0.0
        for i in range(temp_ref_time_step_list_len):
            temp_ref_time_step_list.append(init_step)
            init_step += 0.5

        current_time_step_list = duplicated_droped_df['time'].to_list()
        diff_set = set(temp_ref_time_step_list) - set(current_time_step_list)
        if len(diff_set) != 0:
            res_time_step_list = list(diff_set)
            res_time_step_list_len = len(res_time_step_list)
            res_padding_arr_col = duplicated_droped_df.shape[1] - 1
            res_padding_arr = np.full((res_time_step_list_len, res_padding_arr_col), np.nan)
            res_time_step_arr = np.array(res_time_step_list).reshape(res_time_step_list_len, 1)
            f_res_padding_arr = np.concatenate([res_time_step_arr, res_padding_arr], axis=1)
            f_res_padding_df = pd.DataFrame(data=f_res_padding_arr, columns=duplicated_droped_df.columns)
            res_duplicated_droped_df = pd.concat([duplicated_droped_df, f_res_padding_df], axis=0)
            sorted_res_duplicated_droped_df = res_duplicated_droped_df.sort_values(by='time')
            duplicated_droped_df = sorted_res_duplicated_droped_df

        interpolated_df = duplicated_droped_df.interpolate(method='linear', axis=0)
        aligned_df = interpolated_df[interpolated_df['time'] % 0.5 == 0.0]
        f_aligned_df = aligned_df.set_index('time')
        return f_aligned_df

    def cell_data_append(self,
                         ver_dict: dict,
                         refined_dict: dict,
                         norm_type: str = None):
        """"""
        for temp_data_type in iter(ver_dict):
            if temp_data_type != 'Static':
                temp_verified_cell_data_df = ver_dict[temp_data_type]
                # get cell data and append
                if temp_data_type in ['Charge #1', 'Charge #2', 'Discharge']:
                    temp_cell_voltage_data = temp_verified_cell_data_df['voltage']
                    temp_cell_voltage_data = DataPreProcess.data_normal(temp_cell_voltage_data, norm_type)
                    refined_dict[f'{temp_data_type}-voltage'].append(temp_cell_voltage_data)
                elif temp_data_type in ['Charge #3', 'Charge #4']:
                    # get voltage data
                    temp_cell_voltage_data = temp_verified_cell_data_df['voltage']
                    temp_cell_voltage_data = DataPreProcess.data_normal(temp_cell_voltage_data, norm_type)
                    refined_dict[f'{temp_data_type}-voltage'].append(temp_cell_voltage_data)

                    # get current data
                    temp_cell_current_data = temp_verified_cell_data_df['current']
                    temp_cell_current_data = DataPreProcess.data_normal(temp_cell_current_data, norm_type)
                    refined_dict[f'{temp_data_type}-current'].append(temp_cell_current_data)
                else:
                    raise ValueError

    @staticmethod
    def data_normal(data: pd.Series, norm_type: str = None) -> pd.Series:
        """"""
        if norm_type is not None:
            if norm_type == 'LayerStd':
                temp_mean = data.mean()
                temp_std = data.std()
                temp_cell_voltage_data = (data - temp_mean) / temp_std
            else:
                raise ValueError
        else:
            temp_cell_voltage_data = data
        return temp_cell_voltage_data


if __name__ == '__main__':
    data_file_base = 'D:\\workspace\\battery_dataset\\2600P-01_DataSet\\organized_data'
    curr_file_type = 'pickle'
    data_file_path = os.path.join(data_file_base, curr_file_type)

    # ('Static', 'Charge #1', 'Charge #2', 'Charge #3', 'Discharge', 'Charge #4')
    curr_data_type = ('Charge #1', 'Charge #2', 'Discharge', 'Charge #3')

    # ('LayerNormal', 'LocalMinMax', GlobalMinMax, 'Standardization')
    m_std_type = 'LayerNormal'

    # (OverSampling, UnderSampling)
    # OverSampling: 'Random', 'SMOTE', 'ADASYN'
    # UnderSampling: 'ClusterCentroids', 'Random', 'NearMiss', 'TomekLinks'
    #                'AllKNN ', 'CondensedNearestNeighbour'
    m_resampling_type = ('', '')

    # class init
    m_DataPreProcess = DataPreProcess(data_file_path, curr_data_type)

    # run
    m_DataPreProcess.data_process()

    sys.exit(0)
