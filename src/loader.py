# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loader.py
# Time       ：4/11/2024 3:25 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：several dataloader, 关于数据的放在这里，不同的方法的可以在baseline_utils修改
"""
import torch
from torch.utils.data import DataLoader, Dataset
from baseline_utils import collate_fn_corgan, collate_fn_smart, collate_fn_prism



def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]} # conditions: [B,V,M]

def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_dict):
    """for first, third stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader




# def get_note():
#     pass
#
def collate_fn_note(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]} # 干脆直接在论文里处理。就是model forward多一个部分
    # data['note'] = [[d['note']] * len(d['conditions']) for d in batch] # 把这个note复制到每个condition上，不然存储的太大了。
    return data # conditions: [B,V,M]


def get_special_input(config):
    if config['MODEL']== 'PRISM':
        return collate_fn_prism
    elif config['DATASET'] == 'MIV-Note':
        return collate_fn_note
    else:
        return collate_fn_dict
