""""""
#

#
import torch.nn as nn
import torch.nn.functional as F

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
                 in_dim,
                 num_cls,
                 loss_func=None,
                 dropout=0.1):
        """"""
        super(BaseLine_MCNN, self).__init__()
        self.model_name = self.__class__.__name__
        self.model_batch_out = None
        self.model_batch_loss = None

    def forward(self, x, y=None):
        """"""
        res = x

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