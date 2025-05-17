# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : baseline_utils.py
# Time       ：9/12/2024 10:15 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：baseline utils
"""
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import itertools

####### SMART #######
def filter_dataset(dataset, config):
    """
    filter dataset for SMART
    """
    new_data = []
    # print("Dataset filtering,", dataset[0])
    for patient in dataset:
        incomplete = np.array(patient['incomplete'])
        if np.any(incomplete): # 有True就跳过
            continue
        new_data.append(patient)
    return new_data


def generate_mask(dataset, config, threshold=None):
    """
    generate mask for SMART
    """
    # 这里没啥用，得在里面进行mask
    return dataset


def collate_fn_smart(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]}
    batch_copy = copy.deepcopy(batch) # 不copy会导致m_Condition何condition一样
    data['m_conditions'] = [random_mask_session(d['conditions']) for d in batch_copy]
    data['m_procedures'] = [random_mask_session(d['procedures']) for d in batch_copy]
    data['m_drugs'] = [random_mask_session(d['drugs_hist']) for d in batch_copy]
    return data

def random_mask_session(seq, mask_prob=0.2): # 0.5容易造成有问题
    mask_idx = ['<pad>'] #
    # mask_pad = []
    for i, _ in enumerate(seq):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob: # 这个比例或许可以改一下
            seq[i] = mask_idx
        else:
            pass
    return seq


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=None):
    """for first, third stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader



##### prism #####


def valid_seq(seq):
    """
    valid sequence
    """
    mark = [3]* len(seq)
    for index, data in enumerate(seq):
        if data != '<pad>':
            mark[index] = 0
        else:
            sub_seq = set(list(itertools.chain(*seq[:index])))
            if len(sub_seq)>1:
                result = np.array([sublist == ['<pad>'] for sublist in seq[:index]]) # 最后一位
                last_index = np.where(result == False)[0][-1]
                mark[index] = index-last_index
            else:
                mark[index] = 10 # 这里0代表pad，并且从没出现过, 使用max_len
    return mark

def local_mask(data):
    conditions = data['conditions']
    procedures = data['procedures']
    drugs = data['drugs_hist']
    zip_mark = list(zip(valid_seq(conditions), valid_seq(procedures), valid_seq(drugs))) # [[1,2,1],[1,0,0]]
    return zip_mark


def get_last_visit_sample(samples):
    """提取sample中的最后一次就诊记录"""
    last_visits = {}
    for record in samples:
        patient_id = record['patient_id']
        visit_id = float(record['visit_id'])  # 将visit_id转换为整数, 变为float是一样的，之前是int
        if patient_id not in last_visits or visit_id > float(last_visits[patient_id]['visit_id']):
            last_visits[patient_id] = record
    print("Patient Number: ", len(last_visits))
    return last_visits

def missing_statistics(samples, config=None):
    samples = get_last_visit_sample(samples)
    conditions_miss, procedure_miss, drug_miss, note_miss = 0,0,0,0
    conditions_total, procedure_total, drug_total, note_total = 0,0,0,0
    for _, patient in samples.items():  # 转为visit-level
        visit_length = len(patient['conditions'])
        conditions_total += visit_length
        procedure_total += visit_length
        drug_total += visit_length
        note_total += visit_length
        for visit in range(visit_length):
            conditions = patient['conditions'][visit]
            procedures = patient['procedures'][visit]
            drugs = patient['drugs_hist'][visit]
            if config['DATASET']=='MIV-Note':
                note = patient['note'][visit]
            else:
                note = [0]
            if conditions == ['<unk>']:
                conditions_miss +=1
            if procedures == ['<unk>']:
                procedure_miss += 1
            if drugs == ['<unk>']:
                drug_miss += 1
            if sum(note) ==0:
                note_miss +=1
    note_spe = note_miss / note_total if config['DATASET']=='MIV-Note' else 'NA'
    print("train ratio missing", conditions_miss / conditions_total, procedure_miss/procedure_total, drug_miss/ drug_total, note_spe)


def collate_fn_prism(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]}
    batch_copy = copy.deepcopy(batch) # 不copy会导致m_Condition何condition一样
    data['global'] = [[0.001, 0.15, 0.1] for _ in batch_copy] # global计算 0.9997317596566524 0.8598444206008584 0.9093347639484979
    # local, 这里不要用无穷了，用2算了
    data['local'] = [local_mask(d) for d in batch_copy]
    # print(data['local'])
    # print('XXXX',a)
    return data



##### for ehrgan #####

def random_mask(lis, mask_prob=0.3):
    mask_idx = ['<pad>'] #
    # mask_pad = []
    prob = random.random()
    # mask token with 15% probability
    if prob < mask_prob: # 这个比例或许可以改一下
        lis = mask_idx
    else:
        pass
    return lis
def collate_fn_corgan(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]}
    batch_copy = copy.deepcopy(batch) # 不copy会导致m_Condition何condition一样
    data['m_conditions'] = [random_mask(d['conditions']) for d in batch_copy]
    data['m_procedures'] = [random_mask(d['procedures']) for d in batch_copy]
    data['m_drugs'] = [random_mask(d['drugs']) for d in batch_copy]
    return data

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['conditions'])

    def __getitem__(self, item):
        data = {}
        conditions = self.dataset['conditions'][item]
        procedures = self.dataset['procedures'][item]
        drugs = self.dataset['drugs'][item]
        note = self.dataset['note'][item]
        data['conditions'] = conditions
        data['procedures'] = procedures
        data['drugs'] = drugs
        data['note'] = note
        return data

def reshape_dataset(dataset, config):
    samples = get_last_visit_sample(dataset) # [{}]
    new_dataset = {'conditions': [], 'procedures': [], 'drugs': [], 'note': []}
    for id, patient in samples.items():
        conditions = patient['conditions'] # [[]]
        procedures = patient['procedures']
        drugs = patient['drugs_hist']
        note = patient['note'] if config['DATASET']=='MIV-Note' else [0] * len(conditions)
        for visit in range(len(conditions)):
            new_dataset['conditions'].append(conditions[visit])
            new_dataset['procedures'].append(procedures[visit])
            new_dataset['drugs'].append(drugs[visit])

            new_dataset['note'].append(note[visit])

    new_dataset = CustomDataset(new_dataset)
    return new_dataset



   




if __name__ == '__main__':
    # test note statistics
    from config import config
    from utils import load_pickle

    config['TASK'] = 'PHE'
    config['DATASET'] = 'MIV-Note'
    root_to = '/home/czhaobo/MMHealth/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])
    samples = load_pickle(root_to + 'datasets_pre_stand.pkl')

    missing_statistics(samples, config=config)

