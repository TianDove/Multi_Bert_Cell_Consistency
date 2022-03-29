""""""
#
import math
import random
import copy

#
from tqdm import tqdm
import torch
import torch.nn as nn
#
import utility_function

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
                 rnd_token_table):
        """"""
        super(TokenSubstitution, self).__init__()
        self.batch_size = batch_size
        self.max_seg = max_seg
        self.in_len = in_len
        self.token_dim = token_dim
        self.num_sp_token = 6
        self.rnd_token_table = rnd_token_table

        self.sp_token_embedding = nn.Embedding(num_embeddings=self.num_sp_token,
                                               embedding_dim=in_len,
                                               max_norm=1.0,
                                               padding_idx=0)
        # special token define
        self.sp_token_idx_dict = {
            'PAD': 0,
            'SOS': 1,
            'EOS': 2,
            'STP': 3,
            'CLS': 4,
            'MASK': 5
        }

    def mlm_repalce(self,
                    in_data,
                    mask_rate=0.15,
                    visulize=True):
        """"""
        # get mask token embedding
        mask_token = self.sp_token_embedding(torch.tensor([self.sp_token_idx_dict['MASK']],
                                                          dtype=torch.long)).repeat(self.batch_size, 1, 1)
        # visulize mask position
        if visulize:
            temp_mask_df_list = []

        mlm_positions_and_labels_dict = {}
        for para_name, para_value in iter(in_data.items()):

            mlm_positions_and_labels_dict[para_name] = []
            _temp_para_array = para_value  # B * L * T
            temp_pred_positions_and_labels_list = []

            if visulize:
                import pandas as pd
                temp_para_df = pd.DataFrame(data=_temp_para_array[0, :, :].cpu().detach().numpy())
                mask_type = None

            for t_idx in range(_temp_para_array.shape[1]):
                temp_mask_token = None
                temp_org_token = _temp_para_array[:, t_idx, :].unsqueeze(1)
                temp_pred_positions_and_labels = [t_idx, temp_org_token]  # (position, label)
                if random.random() < mask_rate:
                    if random.random() < 0.8:
                        # 80%的时间：将词替换为“<mask>”词元
                        temp_mask_token = mask_token

                        if visulize:
                            mask_type = 'mask'
                    else:
                        if random.random() < 0.5:
                            # 10%的时间：保持词不变
                            temp_mask_token = temp_org_token

                            if visulize:
                                mask_type = 'org'
                        else:
                            # 10%的时间：用随机词替换该词
                            temp_mask_token = random.choice(self.rnd_token_table).repeat(self.batch_size, 1, 1)

                            if visulize:
                                mask_type = 'rnd'

                    # replace token in the sequence
                    _temp_para_array[:, t_idx, :] = temp_mask_token.squeeze(1)

                    if visulize:
                        temp_para_df.iloc[t_idx, :] = pd.Series(data=temp_mask_token[0, 0, :].cpu().detach().numpy(),
                                                                name=mask_type)
                        temp_para_df = temp_para_df.rename(index={t_idx: mask_type})

                    # append idx and label
                    mlm_positions_and_labels_dict[para_name].append(temp_pred_positions_and_labels)
            if visulize:
                temp_mask_df_list.append(temp_para_df)

        if visulize:
            temp_vis_mask_df = pd.concat(temp_mask_df_list, axis=0)
        return in_data, mlm_positions_and_labels_dict, temp_mask_df_list

    def forward(self,
                in_data,
                positions_and_labels_dict,
                num_cls,
                mask_df_list,
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
        sp_token_embedding_dict = {}
        for sp_token_key, sp_token_idx in iter(self.sp_token_idx_dict.items()):
            temp_sp_token_embedding = self.sp_token_embedding(torch.tensor([sp_token_idx],
                                                                           dtype=torch.long))
            sp_token_embedding_dict[sp_token_key] = temp_sp_token_embedding

        init_segment_index = [0]
        segment_index = []

        temp_sequence = []

        if visulize:
            temp_mask_df_list = []

        for para_name, para_batch in iter(in_data.items()):

            if visulize:
                import pandas as pd
                temp_para_df_list = []

            temp_segment_list = []
            if init_segment_index[0] == 0:
                if num_cls != 0:
                    for i in range(num_cls):
                        temp_segment_list.append(sp_token_embedding_dict['CLS'].repeat(self.batch_size, 1, 1))
                        if visulize:
                            temp_para_df_list.append(
                                pd.DataFrame(data=sp_token_embedding_dict['CLS'].cpu().detach().numpy(),
                                             index=['CLS']))
                else:
                    temp_segment_list.append(sp_token_embedding_dict['CLS'].repeat(self.batch_size, 1, 1))
                    if visulize:
                        temp_para_df_list.append(
                            pd.DataFrame(data=sp_token_embedding_dict['CLS'].cpu().detach().numpy(),
                                         index=['CLS']))

                temp_segment_list.append(sp_token_embedding_dict['SOS'].repeat(self.batch_size, 1, 1))
                if visulize:
                    temp_para_df_list.append(
                        pd.DataFrame(data=sp_token_embedding_dict['SOS'].cpu().detach().numpy(),
                                     index=['SOS']))

            temp_segment_list.append(para_batch)
            if visulize:
                temp_para_df_list.append(mask_df_list[init_segment_index[0]])

            temp_segment_shape = para_batch.shape

            if init_segment_index[0] != (self.max_seg - 1):
                temp_segment_list.append(sp_token_embedding_dict['STP'].repeat(self.batch_size, 1, 1))

                if visulize:
                    temp_para_df_list.append(
                        pd.DataFrame(data=sp_token_embedding_dict['STP'].cpu().detach().numpy(),
                                     index=['STP']))

            else:
                temp_segment_list.append(sp_token_embedding_dict['EOS'].repeat(self.batch_size, 1, 1))

                if visulize:
                    temp_para_df_list.append(
                        pd.DataFrame(data=sp_token_embedding_dict['EOS'].cpu().detach().numpy(),
                                     index=['EOS']))

            # concat tokens
            temp_segment_arr = torch.cat(temp_segment_list, dim=1)

            if visulize:
                temp_para_mask_df = pd.concat(temp_para_df_list, axis=0)
                temp_mask_df_list.append(temp_para_mask_df)
                # generate segment index
            temp_segment_idx = init_segment_index * temp_segment_arr.shape[1]
            segment_index += temp_segment_idx
            init_segment_index[0] += 1

            # append segment arr to sequence list
            temp_sequence.append(temp_segment_arr)

        if visulize:
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


class MyBertEncoder(nn.Module):
    """"""
    def __init__(self,
                 max_num_seg: int,
                 max_num_token: int,
                 embedding_token_dim: int,
                 encoder_para: tuple,):  # (layers, nhead, hidden_size)
        """"""
        super(MyBertEncoder, self).__init__()
        self.token_embedding = CovTokenEmbedding(1)
        self.segment_embedding = nn.Embedding(num_embeddings=max_num_seg,
                                              embedding_dim=embedding_token_dim,
                                              max_norm=1.0, )

        self.position_embedding = nn.Parameter(torch.rand(1, max_num_token, embedding_token_dim))  # (0, 1)

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
        segments_embedding = self.segment_embedding(torch.tensor(segments,
                                                                 dtype=torch.long)).unsqueeze(0)

        seg_pos_embedding = segments_embedding + self.position_embedding[:, 0:len(segments), :]

        seg_pos_embedding = seg_pos_embedding.repeat(batch_size, 1, 1)

        # add position embedding and segment embedding to the data
        out_data = out_data + seg_pos_embedding

        # encoder input B * L * EB
        ecd_out = self.encoder_blk(out_data)

        return ecd_out


class MyMspMaskToken(nn.Module):
    """"""
    def __init__(self,
                 token_tuple: (int, bool, int),  # (token_len, over_lap, step)
                 rnd_token_table,
                 rnd_para_table,
                 batch_size: int,
                 embedding_token_dim: int,
                 max_num_seg: int):
        """"""
        super(MyMspMaskToken, self).__init__()

        # data
        self.batch_size = batch_size
        self.rnd_token_table = torch.from_numpy(rnd_token_table).to(dtype=torch.float32)
        self.num_cls = None

        self.rnd_para_table = {}
        for para_name, para_table in iter(rnd_para_table.items()):
            self.rnd_para_table[para_name] = torch.from_numpy(para_table).to(dtype=torch.float32)

        # layer
        self.tokenizer = utility_function.Tokenizer(token_tuple)

        self.token_substitution = TokenSubstitution(token_tuple[0],
                                                    max_num_seg,
                                                    embedding_token_dim,
                                                    self.batch_size,
                                                    self.rnd_token_table)

        # # MASK LM
        # self.mask_lm = MaskLM(token_tuple[0], embedding_token_dim)
        # self.mask_lm_loss = nn.MSELoss()
        #
        # # next_para_pred
        # self.next_para_pred = NextParameterPred(embedding_token_dim)
        # self.next_para_pred_loss = nn.CrossEntropyLoss()

    def nsp_replace(self,
                    in_data: dict,
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
            rnd_para_data = random.choice(self.rnd_para_table[rnd_para_name]).repeat(self.batch_size, 1)
            in_data[rnd_key] = rnd_para_data

            # 0 - not next para, 1 - next para
            # [true para pos, sub para name, 0/1]
            temp_nsp_label = [rnd_key, rnd_para_name, 0]
        else:
            rnd_para_name = rnd_key
            temp_nsp_label = [rnd_key, rnd_para_name, 1]

        nsp_positions_and_labels_dict[rnd_key] = temp_nsp_label
        rpl_num += 1
        #
        self.num_cls = rpl_num

        return in_data, nsp_positions_and_labels_dict

    def forward(self, in_data):
        """
        :param in_data: dict, {'ch1v': torch.Tensor, # Batch * 161
                               'ch2v': torch.Tensor, # Batch * 177
                               'dcv': torch.Tensor, # Batch * 247
                               'ch3v': torch.Tensor, # Batch * 218
                               'ch3c': torch.Tensor, # Batch * 218}
        :return:
        """
        # nsp replace
        out_data, nsp_positions_and_labels_dict = self.nsp_replace(in_data, self.rnd_para_table)

        # tokenize in_data
        temp_token_dict = {}
        temp_label_arr = []
        for key, value in iter(out_data.items()):
            # B * L
            temp_para_arr = value
            temp_token_list = []
            if key != 'label':
                for row in iter(temp_para_arr):
                    # 1 * L
                    temp_token_arr, num_token = m_tokenizer.tokenize(row.numpy())
                    temp_token_arr = torch.from_numpy(temp_token_arr).to(dtype=torch.float32)

                    # update token list
                    temp_token_list.append(temp_token_arr.unsqueeze(0))

                # update token dict
                # B * n_token * token_len
                temp_token_dict[key] = torch.cat(temp_token_list, dim=0)
            else:
                # update label dict
                # 1 * B
                temp_label_arr = temp_para_arr

        # replace token
        out_data, mlm_positions_and_labels_dict, mask_df_list = self.token_substitution.mlm_repalce(temp_token_dict,
                                                                                                    visulize=True)

        # insect special token embedding
        out_data, segments, trans_positions_and_labels_dict, sub_df = self.token_substitution(out_data,
                                                                                      mlm_positions_and_labels_dict,
                                                                                              self.num_cls,
                                                                                              mask_df_list,
                                                                                              visulize=True)

        return out_data, temp_label_arr, segments, nsp_positions_and_labels_dict, trans_positions_and_labels_dict


class MyMulitBERT(nn.Module):
    """"""
    def __init__(self,
                 token_tuple,
                 rnd_token_table,
                 rnd_para_table,
                 batch_size,
                 embedding_token_dim,
                 max_num_seg,
                 max_token,
                 encoder_para):
        """"""
        super(MyMulitBERT, self).__init__()
        self.model_name = self.__class__.__name__
        self.batch_size = batch_size

        self.parasub_tokenize_tokensub = MyMspMaskToken(token_tuple=token_tuple,
                                                        rnd_token_table=rnd_token_table,
                                                        rnd_para_table=rnd_para_table,
                                                        batch_size=batch_size,
                                                        embedding_token_dim=embedding_token_dim,
                                                        max_num_seg=max_num_seg)

        self.encoder = MyBertEncoder(max_num_seg=max_num_seg,
                                     max_num_token=max_token,
                                     embedding_token_dim=embedding_token_dim,
                                     encoder_para=encoder_para)

        # # MASK LM
        self.mask_lm = MaskLM(token_tuple[0], embedding_token_dim)
        #
        # next_para_pred
        self.next_para_pred = NextParameterPred(embedding_token_dim)

    def forward(self, in_data):
        """"""
        out_data, _, segments, nsp_pred_loc, mlm_pred_loc = self.parasub_tokenize_tokensub(in_data)

        encoder_out = self.encoder(out_data, segments, self.batch_size)

        # mask lm
        mlm_pred = self.mask_lm(encoder_out, mlm_pred_loc)
        #
        # next para prediction
        temp_cls_token = encoder_out[:, 0, :].unsqueeze(1)
        next_para_pred = self.next_para_pred(temp_cls_token)

        return encoder_out, mlm_pred, next_para_pred


if __name__ == '__main__':

    import os
    import sys
    import torch
    import numpy as np
    import utility_function
    import dataset_and_dataloader

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    os.environ['OMP_NUM_THREADS'] = '1'

    # set device
    USE_GPU = False
    if USE_GPU:
        device = utility_function.try_gpu()
    else:
        device = torch.device('cpu')

    # load dataset file
    m_data_file_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'
    m_data_dict = utility_function.read_pickle_file(m_data_file_path)

    # load random token file
    m_rnd_token_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndtoken_32.pickle'
    m_rnd_token = utility_function.read_pickle_file(m_rnd_token_path)

    # load random para file
    m_rnd_para_path = '.\\pik\\2022-03-05-13-36-24_Cell_set_MinMax_pad_labels_rndpara.pickle'
    m_rnd_para = utility_function.read_pickle_file(m_rnd_para_path)

    bsz = 64
    m_data_loader = dataset_and_dataloader.CellDataLoader.creat_data_loader(m_data_dict, 'pretrain', bsz, True, 1, True)

    # calculate token number
    example_input = m_data_dict['pretrain'][42627]
    _token_tuple = (32, False, 1)
    m_tokenizer = utility_function.Tokenizer(_token_tuple)
    num_token_dict = {}
    for _key, _value in iter(example_input.items()):
        _temp_para_arr = _value
        if _key != 'label':
            _temp_token_arr, num_token_dict[_key] = m_tokenizer.tokenize(_temp_para_arr)

    # model init
    m_model = MyMulitBERT(token_tuple=_token_tuple,
                          rnd_token_table=m_rnd_token,
                          rnd_para_table=m_rnd_para,
                          batch_size=bsz,
                          embedding_token_dim=16,
                          max_num_seg=5,
                          max_token=10000,
                          encoder_para=(3, 4, 256))

    for i, data in enumerate(m_data_loader):
        out = m_model(data)
    sys.exit(0)