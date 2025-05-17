# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：4/11/2024 3:24 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：several common tools
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import pickle
import gzip

from sklearn.manifold import TSNE
from pyhealth.medcode.codes.atc import ATC
from pyhealth.medcode import InnerMap
from pyhealth.datasets import SampleBaseDataset
from pyhealth.tokenizer import Tokenizer
from itertools import chain
from typing import Optional, Tuple, Union, List
import itertools
from torch.nn.utils.rnn import pad_sequence


def set_random_seed(seed):
    """ 设置随机种子以确保代码的可重复性 """
    random.seed(seed)       # Python 内置的随机库
    np.random.seed(seed)    # NumPy 库
    torch.manual_seed(seed) # PyTorch 库

    # 如果您使用 CUDA，则还需要添加以下两行代码
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU



def split_by_patient(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        train_ratio=1.0,
        seed: Optional[int] = None,
        warm_cold: bool = False,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys()) # 存储数据 {patientID: [index]}
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    np.random.seed(seed)
    np.random.shuffle(train_patient_indx)
    train_patient_indx = train_patient_indx[: int(len(train_patient_indx) * train_ratio)]
    val_patient_indx = patient_indx[
                       int(num_patients * ratios[0]): int(
                           num_patients * (ratios[0] + ratios[1]))
                       ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])):]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))

    min_length = min(len(lst) for lst in dataset.patient_to_index.values())
    print("最短列表的长度为:", min_length)

    if warm_cold:
        warm_patient_index = []
        cold_patient_index = []
        # 这里放一些东西
        for i in test_patient_indx:
            patient_index = dataset.patient_to_index[i] # lis
            if len(patient_index) > 1: # 最少是1数据来着
                warm_patient_index.extend(patient_index)
            else:
                cold_patient_index.extend(patient_index)
        if warm_cold == 'warm':
            test_dataset = torch.utils.data.Subset(dataset, warm_patient_index)
        elif warm_cold == 'cold':
            test_dataset = torch.utils.data.Subset(dataset, cold_patient_index)
    else:
        test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    # test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        data = pickle.dump(data, f)
    print("File has beeen saved to {}.".format(file_path))
    return



"""adjust_learning_rate"""
def lr_poly(base_lr, iter, max_iter, power):
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, config, max_iter):
    lr = lr_poly(config['LR'], i_iter, max_iter, 0.9) # power=0.9
    optimizer.param_groups[0]['lr'] = np.around(lr,5)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

from pyhealth.models import BaseModel

def get_tokenizers(dataset, special_tokens=False):
    if not special_tokens:
        special_tokens = ["<pad>", "<unk>"] # 把pad取消
    feature_keys = ["conditions", "procedures", "drugs"]
    feature_tokenizers = {}
    for feature_key in feature_keys:
        feature_tokenizers[feature_key] = Tokenizer(
            tokens=dataset.get_all_tokens(key=feature_key),
            special_tokens=special_tokens,
        )
        print(feature_key, feature_tokenizers[feature_key].get_vocabulary_size())
    return feature_tokenizers



def get_last_visit(hidden_states, mask):
    """防止遇到全空的行
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1

        # 创建一个布尔型的掩码，用于标识矩阵中含有 -1 的行
        mask = (last_visit == -1)
        # 将含有 -1 的行用 0 填充，避免报错
        last_visit = torch.where(last_visit == -1, torch.tensor(0).to(mask.device), last_visit)
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        # last_hidden_state = last_hidden_state.masked_fill(mask.unsqueeze(1), 0)
        last_hidden_state[mask] = 0 # 为空

        return last_hidden_state


def pad_list(nest_list, device, max_len=10):
    tensors = [torch.tensor(sublist[-max_len:]) for sublist in nest_list]
    # 使用 pad_sequence 进行填充
    padded_tensor = pad_sequence(tensors, batch_first=True)
    return padded_tensor.to(device)









##### DEPOT
def pad_batch(embs, lengths):
    """reshape a list into a batch"""
    lengths = np.asarray(lengths)
    cum_lengths = np.cumsum(lengths)
    batch_lis = [embs[i - l:i] for i, l in zip(cum_lengths, lengths)]

    batch_lis = torch.nn.utils.rnn.pad_sequence(batch_lis, batch_first=True, padding_value=0) # [torch.randn(4,8), torch.randn(3,8)]->[torch.randn(2,4,8)]
    return batch_lis

def shift_padding(mask):
    shifted_mask = torch.zeros_like(mask)  # 创建一个与原始 mask 矩阵相同形状的零矩阵
    shifted_mask[:, :-1] = mask[:, 1:]  # 将原始 mask 矩阵向左移动一位，右边用零填充
    return shifted_mask.to(mask.device)



def get_indices(adjacency_matrix):
    """邻接矩阵的非零"""
    nonzero_column_indices = torch.nonzero(adjacency_matrix, as_tuple=False)[:, 1]
    lens = list(torch.sum(adjacency_matrix, dim=1).long().cpu().numpy())
    return nonzero_column_indices, lens

def get_nonzero_values(matrix):
    """其他非0 indice"""
    nonzero_values = torch.masked_select(matrix, matrix != 0)
    return nonzero_values

def create_interaction_matrix(interactions, total_drugs):
    B, D = interactions.shape  # 用户数和每个用户的最大交互数
    user_drug_matrix = torch.zeros((B, total_drugs), dtype=torch.bool).to(interactions.device)

    valid_interactions = interactions.clone()
    valid_interactions[interactions == -1] = 0

    user_drug_matrix.scatter_(1, valid_interactions, 1)

    if (interactions == -1).any():
        user_drug_matrix[:, 0] = user_drug_matrix[:, 0] & (interactions != -1)

    return user_drug_matrix


def build_map(b_map, max=None, config=None):
    """根据b-map向b-map_进行插值"""
    batch_size, b_len = b_map.size()
    if max is None:
        max = b_map.max() + 1
    if config['USE_CUDA']:
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0).to(b_map.device)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    b_map_.requires_grad=False
    return b_map_

def pad_nested_list(tokens, pad_value=0):
    """[[[1,0,3],[3,0]],[[1,0,3],[3,0],[4,0,0]]]"""
    # 获取最内层和外层的最大长度
    max_len_inner = max(len(inner) for outer in tokens for inner in outer)
    max_len_outer = max(len(outer) for outer in tokens)

    # 初始化填充值矩阵
    padded_tokens = np.full((len(tokens), max_len_outer, max_len_inner), pad_value, dtype=object)

    # 填充矩阵
    for i, outer in enumerate(tokens):
        for j, inner in enumerate(outer):
            padded_tokens[i, j, :len(inner)] = inner

    return padded_tokens.tolist()



def get_contribution(model,config,num_patient=5):
    """获取贡献"""
    output_cond = model.embeddings['conditions'].weight
    output_proc = model.embeddings['procedures'].weight
    output_drug = model.embeddings['drugs'].weight

    grad_cond = output_cond.grad
    grad_proc = output_proc.grad
    grad_drug = output_drug.grad

    con_cond = torch.sum(torch.abs(grad_cond))
    con_proc = torch.sum(torch.abs(grad_proc))
    con_drug = torch.sum(torch.abs(grad_drug))

    total_con = con_cond + con_proc + con_drug
    normal_cond = con_cond / total_con
    normal_proc = con_proc / total_con
    normal_drug = con_drug / total_con
    print("Contribution of conditions: {:.2f}%".format(normal_cond * 100))
    print("Contribution of procedures: {:.2f}%".format(normal_proc * 100))
    print("Contribution of drugs: {:.2f}%".format(normal_drug * 100))


def f1_cal(pred, ground):
    # 转换为集合用于计算
    pred_set = set(pred)
    ground_set = set(ground)
    # 真正例 (TP): 预测和实际都是正的
    tp = len(pred_set.intersection(ground_set))
    # print("TP", pred_set.intersection(ground_set))
    # 假正例 (FP): 预测为正但实际为负
    fp = len(pred_set - ground_set)
    # print("FP", pred_set - ground_set)
    '''
    [['A02B', 'A03B', 'A06A', 'A12B', 'A12C', 'B01A', 'B05X', 'C07A', 'C09A', 'C10A', 'N02A', 'N02B']] [['B05X', 'B01A', 'A12B', 'C07A', 'A06A', 'C10A', 'N02B', 'A03B', 'C09A', 'N06A', 'A04A', 'C09C']] (0.75, 0.75, 0.75, 3, 3) 0.6


[['A03B', 'A06A', 'A12B', 'A12C', 'B01A', 'B05X', 'C07A', 'C09A', 'C10A', 'N02B']] [['B05X', 'B01A', 'A12B', 'C07A', 'A06A', 'C10A', 'N02B', 'A03B', 'C09A', 'N06A', 'A04A', 'C09C']] (0.9, 0.75, 0.8181818181818182, 1, 3) 0.6923076923076923
    '''
    # 假负例 (FN): 预测为负但实际为正
    fn = len(ground_set - pred_set)
    # print("FN", ground_set - pred_set)
    # 计算精确度和召回率
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # 计算 F1 分数
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score, fp, fn

def jaccard(pred, ground):
    pred_set = set(pred)
    ground_set = set(ground)
    # 计算 Jaccard 相似度
    jaccard = len(pred_set.intersection(ground_set)) / len(pred_set.union(ground_set)) if len(pred_set.union(ground_set)) > 0 else 0
    return jaccard
