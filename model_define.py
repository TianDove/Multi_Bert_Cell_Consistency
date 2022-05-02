""""""
#

#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def cal_conv1d_output_size(L_in, k_sz, pad=0, dilation=1, stride=1):
    """"""
    up_frac = L_in + (2 * pad) - dilation * (k_sz - 1) - 1
    down_frac = stride
    res_frac = up_frac / down_frac
    L_out = res_frac + 1
    return L_out

class Conv_Bn_Relu(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, dropout=0.1):
        """
        in:(N, in_ch, L, E)
        out:(N, out_ch, L, E)
        """
        super(Conv_Bn_Relu, self).__init__()
        self.n_pad = (k_size - 1) // 2
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv1d(in_ch, out_ch, k_size, padding=self.n_pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.acti = nn.ReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn(res)
        res = self.acti(res)
        res = self.dropout(res)
        return res


class Dense_Sigmoid(nn.Module):
    """"""
    def __init__(self, in_dim, out_dim, dropout):
        """"""
        super(Dense_Sigmoid, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """"""
        res = self.dropout(x)
        res = self.linear(res)
        res = self.act(res)
        return res


class ResidualBlock(nn.Module):
    """
    (N, C, L)
    """
    def __init__(self, in_ch, out_ch, k_size, dropout):
        """"""
        super(ResidualBlock, self).__init__()
        self.model_name = self.__class__.__name__
        self.conv1 = Conv_Bn_Relu(in_ch, out_ch, k_size, dropout=dropout)
        self.conv2 = Conv_Bn_Relu(out_ch, out_ch, k_size, dropout=dropout)
        self.conv3 = Conv_Bn_Relu(out_ch, out_ch, k_size, dropout=dropout)
        self.x_conv = Conv_Bn_Relu(in_ch, out_ch, 1, dropout=dropout)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = res + self.x_conv(x)
        return res

class LocalCovBlock(nn.Module):

    def __init__(self, in_ch, out_ch, conv_k_sz, pooling_f, in_sz_list, dropout=0.1):
        """"""
        super(LocalCovBlock, self).__init__()

        self.max_pool_k_sz_list = []
        for in_sz in in_sz_list:
            tmp_conv_out_sz = in_sz - conv_k_sz + 1

            if tmp_conv_out_sz < pooling_f:
                raise ValueError('Conv Kernel Size Must Greater Than Pooling Factor')
            elif tmp_conv_out_sz <=0:
                raise ValueError('Conv Out Size is 0 or Less than 0')
            else:
                pass

            tmp_max_pool_k_sz = tmp_conv_out_sz // pooling_f
            self.max_pool_k_sz_list.append(tmp_max_pool_k_sz)

        self.conv_pool_list = nn.ModuleList()
        for pool_k_sz in self.max_pool_k_sz_list:
            tmp_mod = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=1, kernel_size=conv_k_sz),
                nn.MaxPool1d(kernel_size=pool_k_sz, stride=pool_k_sz),
                nn.ReLU(),
            )
            self.conv_pool_list.append(tmp_mod)
            


    def forward(self, x):
        """x: List[tensor]"""
        out_tensor_list = []
        for id, tensor in enumerate(x):
            tmp_tensor = torch.clone(tensor)
            tmp_out = self.conv_pool_list[id](tmp_tensor)
            print(tmp_out.shape)
            out_tensor_list.append(tmp_out)

        cat_out_tensor = torch.cat(out_tensor_list, dim=-1)
        return torch.clone(cat_out_tensor)


class BaseLine_MLP(nn.Module):
    """"""
    def __init__(self,
                 in_dim,
                 num_cls,
                 loss_func=None,
                 dropout=0.1):
        """"""
        super(BaseLine_MLP, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

        # dense
        self.dense1 = Dense_Sigmoid(in_dim, 512, dropout)
        self.dense2 = Dense_Sigmoid(512, 256, dropout)
        self.dense3 = Dense_Sigmoid(256, 128, dropout)

        # last layer
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(128, num_cls)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_func = loss_func

    def forward(self, x, y=None):
        """"""
        res = self.dense1(x)
        res = self.dense2(res)
        res = self.dense3(res)
        res = self.dropout(res)
        res = self.linear(res)
        res = self.softmax(res)

        self.model_batch_out = res

        if (y is not None) and (self.loss_func is not None):
            ls = self.loss_func(res, y)
            self.model_batch_loss = ls
            return ls
        else:
            if y is not None:
                return (res, y)
            else:
                raise ValueError('Label Data is Empty or None.')

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_save_model(self):
        """"""
        attr_need = self.__dict__
        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_FCN(nn.Module):
    """"""
    def __init__(self,
                 in_dim,
                 num_cls,
                 loss_func=None,
                 dropout=0.1):
        """"""
        super(BaseLine_FCN, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

        # conv layer
        self.conv1 = Conv_Bn_Relu(1, 128, 7, dropout=dropout)
        self.conv2 = Conv_Bn_Relu(128, 256, 5, dropout=dropout)
        self.conv3 = Conv_Bn_Relu(256, 128, 3, dropout=dropout)

        # last layer
        self.linear = nn.Linear(128, num_cls)
        self.softmax = nn.Softmax(dim=1)
        self.loss_func = loss_func

    def forward(self, x, y=None):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        # global pooling
        res = F.adaptive_avg_pool1d(res, 1)
        res = res.squeeze()
        # out layer
        res = self.linear(res)
        res = self.softmax(res)

        self.model_batch_out = res

        if (y is not None) and (self.loss_func is not None):
            ls = self.loss_func(res, y)
            self.model_batch_loss = ls
            return ls
        else:
            if y is not None:
                return (res, y)
            else:
                raise ValueError('Label Data is Empty or None.')

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_save_model(self):
        """"""
        attr_need = self.__dict__
        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_ResNet(nn.Module):
    """"""
    def __init__(self,
                 in_dim,
                 num_cls,
                 loss_func=None,
                 dropout=0.1):
        """"""
        super(BaseLine_ResNet, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

        self.residual1 = ResidualBlock(1, 64, 7, dropout=dropout)
        self.residual2 = ResidualBlock(64, 128, 5, dropout=dropout)
        self.residual3 = ResidualBlock(128, 128, 3, dropout=dropout)
        # out layer
        self.linear = nn.Linear(128, num_cls)
        self.softmax = nn.Softmax(dim=1)
        self.loss_func = loss_func

    def forward(self, x, y=None):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.residual1(x)
        res = self.residual2(res)
        res = self.residual3(res)
        # global pooling
        res = F.adaptive_avg_pool1d(res, 1)
        res = res.squeeze()
        # out layer
        res = self.linear(res)
        res = self.softmax(res)

        self.model_batch_out = res

        if (y is not None) and (self.loss_func is not None):
            ls = self.loss_func(res, y)
            self.model_batch_loss = ls
            return ls
        else:
            if y is not None:
                return (res, y)
            else:
                raise ValueError('Label Data is Empty or None.')

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_save_model(self):
        """"""
        attr_need = self.__dict__
        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_MCNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_cls: int,
                 k_size: tuple = (2, 4, 8, 16),
                 win_size: tuple = (4, 8, 16, 32),
                 conv_k_sz_rate: float = 0.05,
                 p_factor: int = 5,
                 loss_func = None,
                 dropout: float = 0.1):
        """"""
        super(BaseLine_MCNN, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

        self.in_dim = in_dim
        self.num_cls = num_cls
        self.k_tuple = k_size
        self.win_tuple = win_size
        self.loss_func = loss_func
        self.dropout = dropout
        self.p_factor = p_factor
        self.conv_k_sz_rate = conv_k_sz_rate
        self.conv_k_sz = int(in_dim * conv_k_sz_rate)

        self.num_aug_data = len(self.k_tuple) + len(self.win_tuple) + 1

        self.tmp_input_tensor = torch.rand((1, 1, self.in_dim), dtype=torch.float32)
        self.k_sz_list = []
        for k in self.k_tuple:
            temp_in_data = torch.clone(self.tmp_input_tensor)
            temp_k_down_data = F.interpolate(input=temp_in_data, scale_factor=1 / k)
            self.k_sz_list.append(temp_k_down_data.shape[-1])

        self.win_sz_list = []
        for win in self.win_tuple:
            temp_in_data = torch.clone(self.tmp_input_tensor)
            temp_l_move_data = temp_in_data.unfold(-1, win, 1)
            temp_l_avg_data = temp_l_move_data.mean(-1)
            self.win_sz_list.append(temp_l_avg_data.shape[-1])

        self.org_k_win_sz_list = [in_dim, ] + self.k_sz_list + self.win_sz_list


        self.local_conv_list = LocalCovBlock(in_ch=1,
                                             out_ch=1,
                                             conv_k_sz=self.conv_k_sz,
                                             pooling_f = self.p_factor,
                                             in_sz_list=self.org_k_win_sz_list)

        self.conv = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.act1 = nn.ReLU()

        self.flat = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(2048, 256)
        self.act2 = nn.ReLU()
        self.linear2 = nn.Linear(256, num_cls)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, y=None):
        """"""
        x_3d = x.unsqueeze(dim=1)
        for_org = torch.clone(x_3d)

        # down sample
        for_down_sample = torch.clone(x_3d)
        in_data_size = for_down_sample.shape
        down_sample_data_list = []
        for k in self.k_tuple:
            temp_in_data = torch.clone(for_down_sample)
            temp_k_down_data = F.interpolate(input=temp_in_data, scale_factor=1 / k)
            down_sample_data_list.append(temp_k_down_data)

        # move average
        for_multi_freq = torch.clone(x_3d)
        move_average_data_list = []
        for sz in self.win_tuple:
            temp_in_data = torch.clone(for_multi_freq)
            temp_l_move_data = temp_in_data.unfold(-1, sz, 1)
            temp_l_avg_data = temp_l_move_data.mean(-1)
            move_average_data_list.append(temp_l_avg_data)

        tmp_tensor_list = [for_org, ] + down_sample_data_list + move_average_data_list

        inter_res = self.local_conv_list(tmp_tensor_list)
        inter_res = self.conv(inter_res)
        inter_res = self.pool(inter_res)
        inter_res = self.act1(inter_res)

        inter_res = self.flat(inter_res)
        inter_res = self.linear1(inter_res)
        inter_res = self.act2(inter_res)
        inter_res = self.linear2(inter_res)
        res = self.softmax(inter_res)

        self.model_batch_out = res

        if (y is not None) and (self.loss_func is not None):
            ls = self.loss_func(res, y)
            self.model_batch_loss = ls
            return ls
        else:
            if y is not None:
                return (res, y)
            else:
                raise ValueError('Label Data is Empty or None.')

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_save_model(self):
        """"""
        attr_need = self.__dict__
        save_model_dict = {}
        for attr in attr_need:
            save_model_dict[attr] = getattr(self, attr)
        return save_model_dict

    def set_model_attr(self, model_attr_dict):
        """"""
        for attr, val in model_attr_dict.items():
            setattr(self, attr, val)

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model