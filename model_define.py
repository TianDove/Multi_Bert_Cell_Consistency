""""""
#
import math
import random
import copy

#
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
#
import utility_function


class Conv_Bn_Relu(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, dropout):
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
        res = self.dropout(x)
        res = self.conv1(res)
        res = self.bn(res)
        res = self.acti(res)
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
            return res

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

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
            return res

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

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
            return res

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model

def mytokenize(in_data,
               tokenizer):
    """"""
    # tokenize in_data
    temp_token_dict = {}
    temp_label_arr = []
    for key, value in iter(in_data.items()):
        # B * L
        temp_para_arr = value
        temp_token_list = []
        if key != 'label':
            for row in iter(temp_para_arr):
                # 1 * L
                temp_token_arr, _ = tokenizer.tokenize(row.unsqueeze(0))
                # update token list
                temp_token_list.append(temp_token_arr.unsqueeze(0))

            # update token dict
            # B * n_token * token_len
            temp_token_dict[key] = torch.cat(temp_token_list, dim=0)
        else:
            # update label dict
            # 1 * B
            temp_label_arr = temp_para_arr

    return temp_token_dict, temp_label_arr


def nsp_replace(in_data: dict,
                bsz,
                rnd_para_table,
                max_rpl=1,
                nsp_rate=0.5,
                visulize=True):
    """"""
    rpl_num = 0
    nsp_positions_and_labels_dict = {}
    key_list = list(in_data.keys())
    key_back = copy.deepcopy(key_list)

    key_list.remove('ch1v')
    key_list.remove('label')
    rnd_key = random.choice(key_list)

    if random.random() < nsp_rate:

        # get para key
        key_back.remove(rnd_key)
        key_back.remove('ch1v')
        key_back.remove('label')

        # get para for replace
        rnd_para_name = random.choice(key_back)
        rnd_para_data = random.choice(rnd_para_table[rnd_para_name]).reshape(1, -1).repeat(bsz, 1)
        in_data[rnd_key] = rnd_para_data

        # 0 - not next para, 1 - next para
        # [true para pos, sub para name, 0/1]
        temp_nsp_label = [rnd_key, rnd_para_name, 0]
    else:
        rnd_para_name = rnd_key
        temp_nsp_label = [rnd_key, rnd_para_name, 1]

    nsp_positions_and_labels_dict[rnd_key] = temp_nsp_label
    rpl_num += 1

    return in_data, nsp_positions_and_labels_dict


# define token embedding
class CovTokenEmbedding(nn.Module):
    """"""
    def __init__(self, max_len, dropout=0.0):
        """"""
        super(CovTokenEmbedding, self).__init__()
        self.in_ch = max_len
        self.stride = (1, 1, 1, 1, 1, 1)
        self.ksz = (7, 5, 3, 3, 2, 2)
        self.cov_blk = nn.Sequential()
        for i in range(len(self.stride)):
            self.cov_blk.add_module(f'conv{i}',
                                    nn.Conv1d(in_channels=self.in_ch,
                                              out_channels=self.in_ch,
                                              kernel_size=self.ksz[i],
                                              stride=self.stride[i],
                                              padding=0))

            self.cov_blk.add_module(f'drop{i}', nn.Dropout(dropout))

    def forward(self, in_data):
        """
        in_data: B * 1 * T
        :param in_data:
        :return:
        """
        out_data = self.cov_blk(in_data)
        return out_data


class NextParameterPred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, embedding_token_dim):
        super(NextParameterPred, self).__init__()
        self.linear1 = nn.Linear(embedding_token_dim, embedding_token_dim // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embedding_token_dim // 2)
        self.linear2 = nn.Linear(embedding_token_dim // 2, 2)
        self.outlayer = nn.Softmax(dim=-1)

    def forward(self, in_data):
        # in_data: [batch_size, embedding_token_dim]
        out_data = self.linear1(in_data)
        out_data = self.act(out_data)
        out_data = self.norm(out_data)
        out_data = self.linear2(out_data)
        out_data = self.outlayer(out_data)
        return out_data


class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, in_len, embedding_token_dim):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_token_dim, embedding_token_dim // 2),
                                 nn.GELU(),
                                 nn.LayerNorm(embedding_token_dim // 2),
                                 nn.Linear(embedding_token_dim // 2, in_len))

    def forward(self, X, pred_positions):
        temp_token_for_pred = []
        for idx in iter(pred_positions):
            pos = idx[0]
            temp_token = X[:, pos, :].unsqueeze(1)
            temp_token_for_pred.append(temp_token)
        temp_token_for_pred = torch.cat(temp_token_for_pred, dim=1)

        out_data = self.mlp(temp_token_for_pred)
        return out_data


class TokenSubstitution(nn.Module):
    """"""
    def __init__(self,
                 in_len: int,
                 max_seg: int,
                 token_dim: int,
                 batch_size: int,
                 rnd_token_table,
                 device):
        """"""
        super(TokenSubstitution, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.max_seg = max_seg
        self.in_len = in_len
        self.token_dim = token_dim
        self.num_sp_token = 6
        self.rnd_token_table = rnd_token_table

        self.sp_token_embedding = nn.Embedding(num_embeddings=self.num_sp_token,
                                               embedding_dim=in_len,
                                               padding_idx=0)
        # special token define
        # 'PAD': 0,
        # 'SOS': 1,
        # 'EOS': 2,
        # 'STP': 3,
        # 'CLS': 4,
        # 'MASK': 5
        self.sp_token_id = []
        self.sp_token_tensor = []

    def mlm_token_repalce(self,
                          in_data,
                          mask_rate=0.15,
                          visulize=True):
        """"""
        # get mask token embedding
        mask_token = self.sp_token_tensor[:, 5, :]

        temp_mask_df_list = []
        mlm_positions_and_labels_dict = {}

        for para_name, para_value in iter(in_data.items()):
            mlm_positions_and_labels_dict[para_name] = []
            temp_para_df = pd.DataFrame(data=para_value[0, :, :].cpu().detach().numpy())

            for t_idx in range(para_value.shape[1] - 1):
                temp_mask_token = None
                mask_type = None
                temp_org_token = para_value[:, t_idx, :]
                temp_pred_positions_and_labels = [t_idx, temp_org_token]  # (position, label)
                if random.random() < mask_rate:
                    if random.random() < 0.8:
                        # 80%的时间：将词替换为“<mask>”词元
                        temp_mask_token = mask_token
                        mask_type = 'mask'
                    else:
                        if random.random() < 0.5:
                            # 10%的时间：保持词不变
                            temp_mask_token = temp_org_token
                            mask_type = 'org'
                        else:
                            # 10%的时间：用随机词替换该词
                            temp_mask_token = random.choice(self.rnd_token_table).repeat(self.batch_size, 1)
                            mask_type = 'rnd'

                    # replace token in the sequence
                    para_value[:, t_idx, :] = temp_mask_token
                    temp_para_df.iloc[t_idx, :] = pd.Series(data=temp_mask_token[0, :].cpu().detach().numpy(),
                                                            name=mask_type)
                    temp_para_df = temp_para_df.rename(index={t_idx: mask_type})

                    mlm_positions_and_labels_dict[para_name].append(temp_pred_positions_and_labels)

            # append para df
            temp_mask_df_list.append(temp_para_df)

        # concat all para df
        temp_vis_mask_df = pd.concat(temp_mask_df_list, axis=0)

        return in_data, mlm_positions_and_labels_dict, temp_mask_df_list

    def mlm_sp_toekn_insert(self,
                            in_data,
                            mask_df_list,
                            positions_and_labels_dict,
                            visulize=True):
        """

                :param in_data: dict, {'ch1v': torch.Tensor, # Batch * num_token * token_dim
                                       'ch2v': torch.Tensor, # Batch * num_token * token_dim
                                       'dcv': torch.Tensor, # Batch * num_token * token_dim
                                       'ch3v': torch.Tensor, # Batch * num_token * token_dim
                                       'ch3c': torch.Tensor, # Batch * num_token * token_dim}
                :return:out_data: torch.Tensor, Batch * [len(CLS) + len(SOS) +
                                                         161 + len(STP) +
                                                         177 + len(STP) +
                                                         247 + len(STP) +
                                                         218 + len(STP) +
                                                         218 + len(EOS)]
                """
        init_segment_index = [0]
        segment_index = []
        temp_sequence = []

        # special token
        # 'PAD': 0,
        # 'SOS': 1,
        # 'EOS': 2,
        # 'STP': 3,
        # 'CLS': 4,
        # 'MASK': 5
        sos_token = self.sp_token_tensor[:, 1, :].unsqueeze(1)
        eos_token = self.sp_token_tensor[:, 2, :].unsqueeze(1)
        stp_token = self.sp_token_tensor[:, 3, :].unsqueeze(1)
        cls_token = self.sp_token_tensor[:, 4, :].unsqueeze(1)

        # sp token df
        temp_mask_df_list = []
        sp_token_df = pd.DataFrame(data=self.sp_token_tensor[0, :, :].cpu().detach().numpy(),
                                   index=['PAD', 'SOS', 'EOS', 'STP', 'CLS', 'MASK'])

        for para_name, para_batch in iter(in_data.items()):
            temp_para_df_list = []
            temp_segment_list = []
            if init_segment_index[0] == 0:
                temp_segment_list.append(cls_token)  # CLS
                temp_para_df_list.append(sp_token_df.loc[['CLS']])

                temp_segment_list.append(sos_token)
                temp_para_df_list.append(sp_token_df.loc[['SOS']])

            temp_segment_list.append(para_batch)
            temp_para_df_list.append(mask_df_list[init_segment_index[0]])

            if init_segment_index[0] != (self.max_seg - 1):
                temp_segment_list.append(stp_token)
                temp_para_df_list.append(sp_token_df.loc[['STP']])
            else:
                temp_segment_list.append(eos_token)
                temp_para_df_list.append(sp_token_df.loc[['EOS']])

            # concat tokens
            temp_segment_arr = torch.cat(temp_segment_list, dim=1)
            # append segment arr to sequence list
            temp_sequence.append(temp_segment_arr)

            temp_para_mask_df = pd.concat(temp_para_df_list, axis=0)
            temp_mask_df_list.append(temp_para_mask_df)

            # generate segment index
            temp_segment_idx = init_segment_index * temp_segment_arr.shape[1]
            segment_index += temp_segment_idx
            init_segment_index[0] += 1

        temp_para_mask_df_all = pd.concat(temp_mask_df_list, axis=0)

        temp_pos_label_list = []
        for key, val in iter(positions_and_labels_dict.items()):
            for i in iter(val):
                temp_pos_label_list.append(i)

        index_list = temp_para_mask_df_all.index.tolist()

        idx = 0
        for i, index in enumerate(index_list):
            if index in ['mask', 'rnd', 'org']:
                temp_pos_label_list[idx][0] = i
                idx += 1

        return torch.cat(temp_sequence, dim=1), segment_index, temp_pos_label_list, temp_para_mask_df_all

    def forward(self,
                in_data):
        """

        :param in_data: dict, {'ch1v': torch.Tensor, # Batch * num_token * token_dim
                               'ch2v': torch.Tensor, # Batch * num_token * token_dim
                               'dcv': torch.Tensor, # Batch * num_token * token_dim
                               'ch3v': torch.Tensor, # Batch * num_token * token_dim
                               'ch3c': torch.Tensor, # Batch * num_token * token_dim}
        :return:out_data: torch.Tensor, Batch * [len(CLS) + len(SOS) +
                                                 161 + len(STP) +
                                                 177 + len(STP) +
                                                 247 + len(STP) +
                                                 218 + len(STP) +
                                                 218 + len(EOS)]
        """
        # special token define
        # 'PAD': 0,
        # 'SOS': 1,
        # 'EOS': 2,
        # 'STP': 3,
        # 'CLS': 4,
        # 'MASK': 5

        # get sp token embedding
        self.sp_token_id = torch.tensor([0, 1, 2, 3, 4, 5],
                                        dtype=torch.long,
                                        device=self.device)
        self.sp_token_tensor = self.sp_token_embedding(self.sp_token_id).repeat(self.batch_size, 1, 1)

        out_data, mlm_pos_and_labels_dict, temp_mask_df_list = self.mlm_token_repalce(in_data)

        out_data, segment_index, pos_label_list, mask_df_all = self.mlm_sp_toekn_insert(out_data,
                                                                                        temp_mask_df_list,
                                                                                        mlm_pos_and_labels_dict)

        segment_index_device = torch.tensor(segment_index,
                                            dtype=torch.long,
                                            device=self.device)
        return out_data, segment_index_device, pos_label_list


class MyBertEncoder(nn.Module):
    """"""
    def __init__(self,
                 max_num_seg: int,
                 max_num_token: int,
                 embedding_token_dim: int,
                 encoder_para: tuple,
                 device):  # (layers, nhead, hidden_size)
        """"""
        super(MyBertEncoder, self).__init__()
        self.device = device
        self.scal = math.sqrt(embedding_token_dim)

        self.token_embedding = CovTokenEmbedding(1)
        self.segment_embedding = nn.Embedding(num_embeddings=max_num_seg,
                                              embedding_dim=embedding_token_dim)

        self.position_embedding = nn.Parameter(torch.randn(1, max_num_token, embedding_token_dim))  # (0, 1)

        self.encoder_blk = nn.Sequential()
        for i in range(encoder_para[0]):
            self.encoder_blk.add_module(f'encoder{i}', nn.TransformerEncoderLayer(d_model=embedding_token_dim,
                                                                                  nhead=encoder_para[1],
                                                                                  dim_feedforward=encoder_para[2],
                                                                                  activation='gelu',
                                                                                  batch_first=True))

    def forward(self,
                in_data,
                segments,
                batch_size):  # in_data: (batch_size, num_token, in_len)
        """"""

        # token embedding
        token_embedding_list = []
        for token_idx in range(in_data.shape[1]):
            temp_token_embedding = self.token_embedding(in_data[:, token_idx, :].unsqueeze(1))
            token_embedding_list.append(temp_token_embedding)
        out_data = torch.cat(token_embedding_list, dim=1)

        # get segments embedding
        segments_embedding = self.segment_embedding(segments)

        seg_pos_embedding = segments_embedding + self.position_embedding[:, 0:len(segments), :]

        seg_pos_embedding = seg_pos_embedding.repeat(batch_size, 1, 1)

        # add position embedding and segment embedding to the data
        out_data = out_data + (self.scal * seg_pos_embedding)

        # encoder input B * L * EB
        ecd_out = self.encoder_blk(out_data)

        return ecd_out


class MyMulitBERTPreTrain(nn.Module):
    """"""
    def __init__(self,
                 token_tuple,
                 rnd_token_table,
                 batch_size,
                 embedding_token_dim,
                 max_num_seg,
                 max_token,
                 encoder_para,
                 loss_fun,
                 device):
        """"""
        super(MyMulitBERTPreTrain, self).__init__()
        self.model_name = self.__class__.__name__
        self.batch_size = batch_size
        self.device = device

        self.rnd_token_table = utility_function.read_pickle_file(rnd_token_table)
        self.rnd_token_table = torch.from_numpy(self.rnd_token_table).to(dtype=torch.float32)
        self.rnd_token_table = self.rnd_token_table.to(device)

        self.model_batch_out = None
        self.model_batch_loss = None

        self.nsp_loss = None
        self.mlm_loss = None

        self.token_substitution = TokenSubstitution(in_len=token_tuple[0],
                                                    max_seg=max_num_seg,
                                                    token_dim=embedding_token_dim,
                                                    batch_size=batch_size,
                                                    rnd_token_table=self.rnd_token_table,
                                                    device=device)

        self.encoder = MyBertEncoder(max_num_seg=max_num_seg,
                                     max_num_token=max_token,
                                     embedding_token_dim=embedding_token_dim,
                                     encoder_para=encoder_para,
                                     device=device)

        # # MASK LM
        self.mask_lm = MaskLM(token_tuple[0], embedding_token_dim)
        #
        # next_para_pred
        self.next_para_pred = NextParameterPred(embedding_token_dim)

        self.mlm_loss_func = loss_fun['mlm_loss']
        self.nsp_loss_func = loss_fun['nsp_loss']

    def forward(self, in_data, _, nsp_label):
        """"""
        out_data, segment_index_device, mlm_pos_label_list = self.token_substitution(in_data)

        encoder_out = self.encoder(out_data, segment_index_device, self.batch_size)

        self.model_batch_out = encoder_out

        # mask lm
        mlm_pred = None
        if mlm_pos_label_list != []:
            mlm_pred = self.mask_lm(encoder_out, mlm_pos_label_list)
        #
        # next para prediction
        temp_cls_token = encoder_out[:, 0, :].unsqueeze(1)
        nsp_pred = self.next_para_pred(temp_cls_token)

        if mlm_pred is None:
            mlm_ls = torch.tensor([0.0]).to(torch.float32).to(device=self.device)
        else:
            # mlm loss
            mlm_lab_val = [x[1].unsqueeze(1) for x in mlm_pos_label_list]
            mlm_lab_val_tensor = torch.cat(mlm_lab_val, dim=1)
            mlm_ls = self.mlm_loss_func(mlm_pred, mlm_lab_val_tensor)

        nsp_ls = self.nsp_loss_func(nsp_pred.squeeze(1), nsp_label)

        self.nsp_loss = nsp_ls
        self.mlm_loss = mlm_ls
        self.model_batch_loss = mlm_ls + nsp_ls

        return self.model_batch_loss

    def get_out(self):
        """"""
        return self.model_batch_out

    def get_loss(self):
        """"""
        return self.model_batch_loss

    def get_nsp_loss(self):
        """"""
        return self.nsp_loss

    def get_mlm_loss(self):
        """"""
        return self.mlm_loss

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class MyDownStreamHead(nn.Module):
    """"""
    def __init__(self,
                 tokensub,
                 encoder,
                 num_class,
                 num_token,
                 embedding_token_dim):
        """"""
        super(MyDownStreamHead, self).__init__()

        # for test
        self.tokensub = tokensub
        self.encoder = encoder
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=num_token * embedding_token_dim, out_features=num_class)
        self.activ = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, in_data, seg_index, batch_size):
        """

        :param in_data: (Batch_size, num_token, embedding dim)
        :return:(Batch_size, num_Class)
        """
        in_data = self.encoder(in_data, seg_index, batch_size)
        in_data = self.flat(in_data)
        in_data = self.linear(in_data)
        in_data = self.activ(in_data)
        in_data = self.softmax(in_data)
        return in_data




    # if __name__ == '__main__':
#
#     import os
#     import sys
#     import torch
#     import numpy as np
#     import utility_function
#     import dataset_and_dataloader
#
#     RANDOM_SEED = 42
#     np.random.seed(RANDOM_SEED)
#     torch.manual_seed(RANDOM_SEED)
#
#     os.environ['OMP_NUM_THREADS'] = '1'
#
#     # set device
#     USE_GPU = True
#     if USE_GPU:
#         device = utility_function.try_gpu()
#     else:
#         device = torch.device('cpu')
#
#     # load dataset file
#     m_data_file_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
#     m_data_dict = utility_function.read_pickle_file(m_data_file_path)
#
#     # load random token file
#     m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
#     m_rnd_token = utility_function.read_pickle_file(m_rnd_token_path)
#     m_rnd_token = torch.from_numpy(m_rnd_token).to(torch.float32)
#
#     # load random para file
#     m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'
#     m_rnd_para = utility_function.read_pickle_file(m_rnd_para_path)
#     m_rnd_para_table = {}
#     for para_name, para_table in iter(m_rnd_para.items()):
#         m_rnd_para_table[para_name] = torch.from_numpy(para_table).to(dtype=torch.float32)
#
#     bsz = 64
#     m_data_loader = dataset_and_dataloader.CellDataLoader.creat_data_loader(m_data_dict, 'pretrain', bsz, True, 1, True)
#
#     # calculate token number
#     example_input = m_data_dict['pretrain'][42627]
#     _token_tuple = (32, False, 1)
#     m_tokenizer = utility_function.Tokenizer(_token_tuple)
#
#     # calculate num of tokens
#     num_token = 0
#     first_data = m_data_dict['pretrain'][42627]
#     for para_name, para_data in iter(first_data.items()):
#         if para_name != 'label':
#             temp_data = torch.from_numpy(para_data.reshape(1, -1)).to(dtype=torch.float32)
#             temp_token_data, num_of_token = m_tokenizer.tokenize(temp_data)
#             num_token += num_of_token

    # pretrian

    # # model init
    # m_model = MyMulitBERT(token_tuple=_token_tuple,
    #                       rnd_token_table=m_rnd_token,
    #                       batch_size=bsz,
    #                       embedding_token_dim=16,
    #                       max_num_seg=5,
    #                       max_token=10000,
    #                       encoder_para=(3, 4, 256),
    #                       device=device)
    # m_model.to(device)

    # for i, data in enumerate(m_data_loader):
    #     m_out_data, m_nsp_labels = nsp_replace(data,
    #                                            bsz,
    #                                            m_rnd_para_table)
    #
    #     temp_token_dict, temp_label_arr = mytokenize(m_out_data, m_tokenizer)
    #
    #     temp_token_dict = utility_function.tensor_dict_to_device(temp_token_dict, device)
    #
    #     out = m_model(temp_token_dict)



    # finetune
    # load_model_dict = torch.load('.\\log\\MyMulitBERT_20220331-133439\\MyMulitBERT_20220331-133439_0.pt')
    # load_model = load_model_dict['model'].encoder
    #
    # down_stream_head = MyDownStreamHead(num_class=8,
    #                                     num_token=num_token,
    #                                     embedding_token_dim=16)
    # down_stream_head = utility_function.init_model(down_stream_head).to(device)
    # loss = torch.nn.CrossEntropyLoss()
    #
    # for i, data in enumerate(m_data_loader):
    #     temp_token_dict, temp_label_arr = mytokenize(data, m_tokenizer)
    #     token_tensor_list = []
    #     seg_index_device = []
    #     seg_init_index = 0
    #     for para_name, para_table in iter(temp_token_dict.items()):
    #         temp_num_token = para_table.shape[1]
    #         temp_para_seg_tensor = torch.ones(temp_num_token) * seg_init_index
    #         token_tensor_list.append(para_table)
    #         seg_index_device.append(temp_para_seg_tensor)
    #         seg_init_index += 1
    #
    #     seg_index_tensor = torch.cat(seg_index_device, dim=0).to(torch.long).to(device)
    #     token_tensor = torch.cat(token_tensor_list, dim=1).to(torch.float32).to(device)
    #
    #     ec_out = load_model(token_tensor, seg_index_tensor, bsz)
    #     out = down_stream_head(ec_out)
    #
    #     # label transform
    #     out_sz = out.shape
    #     temp_label_list = []
    #     for label in iter(temp_label_arr):
    #         temp_idx = label.item()
    #         temp_zero_label = torch.zeros([1, out_sz[1]])
    #         temp_zero_label[0, temp_idx] = 1.0
    #         temp_label_list.append(temp_zero_label)
    #     temp_label_tensor = torch.cat(temp_label_list, dim=0).to(torch.float32)
    #
    #     loss_val = loss(out.cpu(), temp_label_tensor)
    #
    # sys.exit(0)