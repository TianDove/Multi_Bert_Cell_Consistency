""""""
#
#
import torch
import torch.nn as nn
#


class CovEmbedding(nn.Module):
    """"""

    def __init__(self):
        """"""
        super(CovEmbedding, self).__init__()
        self.Cov0 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=5)
        self.Cov1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.Cov2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2)

    # in_data:N * C * L
    def forward(self, in_data) -> torch.Tensor:
        """"""
        in_data = self.Cov0(in_data)
        print(in_data.shape)
        # in_data = self.Cov1(in_data)
        # in_data = self.Cov2(in_data)
        out_data = in_data
        return out_data


if __name__ == '__main__':
    import sys

    import pandas as pd

    import utility_function

    loaded_data_list_dict = utility_function.read_pickle_file('.\\pik\\cell_data_list.pickle')
    m_model = CovEmbedding()

    for temp_data_type in loaded_data_list_dict.keys():
        temp_data_type_list = loaded_data_list_dict[temp_data_type]

        #concat data into df
        temp_all_in_df = pd.concat(temp_data_type_list, axis=1)

        # out lists
        out_list = []
        for temp_cell_data in temp_data_type_list:
            temp_tensor = torch.tensor(temp_cell_data.values, dtype=torch.float)
            print(temp_tensor.shape)
            trans_temp_tensor = temp_tensor.expand(1, 1, -1)
            target = m_model(trans_temp_tensor)
            print(target.shape)

            out = target.detach().squeeze(0).squeeze(0)
            out_list.append(pd.Series(out))

        # concat out into df
        temp_out_in_df = pd.concat(out_list, axis=1)
        p = 1
    sys.exit(0)