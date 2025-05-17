# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data.py
# Time       ：4/11/2024 3:25 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：several task definition
"""

import yaml
import os
import gzip
import shutil
import numpy as np

from pyhealth.data import Patient, Visit
from config import config
from pyhealth.datasets import SampleEHRDataset, SampleBaseDataset
from pyhealth.datasets.utils import list_nested_levels
from typing import Dict, List
from pyhealth.datasets import split_by_patient

from note_pre import get_note
from baseline_utils import get_last_visit_sample

from utils import save_pickle
import pandas as pd
from datetime import datetime,timedelta

def re_generate_dataset(samples, seed):
    sample_dataset = SampleEHRDataset(
        samples,
        dataset_name=config['DATASET'],
        task_name=config['TASK'],
    )

    #  split & save dataset
    train_dataset,_, test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=1.0,  # Train test split
        seed=seed,
    )
    print("Regerenate dataset done!")
    return train_dataset, test_dataset




def convert_dataset(samples, dataset_name=None, task_name=None, valid=True, all=False):
    """避免繁琐的处理"""
    if valid:
        return SampleEHRDataset(
                        samples,
                        dataset_name=dataset_name,
                        task_name=task_name,
                    )
    else:
        return SampleEHRDatasetSIMPLE(
                        samples,
                        dataset_name=dataset_name,
                        task_name=task_name,
                        all=all
                    )


class SampleEHRDatasetSIMPLE(SampleBaseDataset):
    def __init__(self, samples: List[str], code_vocs=None, dataset_name="", task_name="", all=False):
        super().__init__(samples, dataset_name, task_name)
        self.samples = samples
        if all:
            self.input_info: Dict = self._validate() # 别的不需要valid，大大减少时间


    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        keys = self.samples[0].keys()
        return list(keys)

    def _validate(self) -> Dict:
        """ 1. Check if all samples are of type dict. """
        keys = self.samples[0].keys()

        """
        4. For each key, check if it is either:
            - a single value
            - a single vector
            - a list of codes
            - a list of vectors
            - a list of list of codes
            - a list of list of vectors
        Note that a value is either float, int, or str; a vector is a list of float
        or int; and a code is str.
        """
        # record input information for each key
        input_info = {}
        for key in keys:
            """
            4.1. Check nested list level: all samples should either all be
            - a single value (level=0)
            - a single vector (level=1)
            - a list of codes (level=1)
            - a list of vectors (level=2)
            - a list of list of codes (level=2)
            - a list of list of vectors (level=3)
            """
            levels = set([list_nested_levels(s[key]) for s in self.samples[:5]]) # 只取前5个判断足够

            level = levels.pop()[0]

            # flatten the list
            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [i for s in self.samples for i in s[key]]
            elif level == 2:
                flattened_values = [j for s in self.samples for i in s[key] for j in i]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]

            """
            4.2. Check type: the basic type of each element should be float,
            int, or str.
            """
            types = set([type(v) for v in flattened_values[:5]]) # 只取前5个判断足够
            type_ = types.pop()
            """
            4.3. Combined level and type check.
            """
            if level == 0:
                # a single value
                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:
                # a single vector or a list of codes
                if type_ in [float, int]:
                    # a single vector
                    lens = set([len(s[key]) for s in self.samples])
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:
                    # a list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:
                # a list of vectors or a list of list of codes
                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])
                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:
                    # a list of list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 3}
            else:
                # a list of list of vectors
                lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}

        return input_info

    def __len__(self):
        return len(self.samples)


def create_label_for_phenotyping(path):
    """
    Create labels for phenotyping task
    """
    path = path + 'mimic-hcup_ccs_2015_definitions.yaml'
    with open(path) as definitions_file:
        definitions = yaml.load(definitions_file, Loader = yaml.FullLoader)

    code_to_group = {}
    for group in definitions:
        codes = definitions[group]['codes']
        for code in codes:
            if code not in code_to_group:
                code_to_group[code] = group # 'V103': 'Tuberculosis'
            else:
                assert code_to_group[code] == group

    id_to_group = sorted(definitions.keys())
    group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group)) # coarse-grained label

    return code_to_group, group_to_id, id_to_group


def get_visit_phenotype(code_to_group, diagnose_visit: List[str]):
    cur_labels = []
    for diagnose in diagnose_visit:
        if diagnose not in code_to_group:
            continue
        group = code_to_group[diagnose]
        cur_labels.append(group)
    cur_labels = list(set(cur_labels))
    return cur_labels



path = '/home/czhaobo/MMHealth/data/'
code_to_group, group_to_id, id_to_group = create_label_for_phenotyping(path)

# basline别的数据集注释掉if，不然报错。
note_dict = get_note(config, "/home/czhaobo/HyperHealth/data/physionet.org/files/mimic-iv-note/2.2/") if config['DATASET'] =='MIV-Note' else None



def extract_patient_note(patient_id, visit_id):
    zero_tenor = [1.0] * 768 # 这个768可以根据不同的PLM进行调整。
    if (patient_id, visit_id) in note_dict.index: # 这个需要宠幸跑
        # 访问索引并获取 C 列的值
        return note_dict.loc[(patient_id, visit_id), 'note_emb']
    else:
        # 如果没找到，返回 768 维的零向量
        return zero_tenor

def filter_lists(A, B, C, label): # 对retain-grasp这些baseline，这里用unk而不是pad
    # Check if D is empty
    if isinstance(label, list) and len(label) == 0:
        return None  # or return empty lists if preferred
    # Check if A, B, and C are all empty
    if not any([A, B, C]):  # This checks if all lists are empty
        return None  # or return empty lists if preferred
    # Add 'mask' to empty lists
    mask = [0, 0, 0]
    if not A:
        A.append('<unk>')
        mask[0] = 1 # 为True就是mask掉的
    if not B:
        B.append('<unk>')
        mask[1] = 1
    if not C:
        C.append('<unk>')
        mask[2] = 1
    return A, B, C, label, mask

####### Phe experiment
def phe_prediction_miii_fn(patient: Patient):
    """
    处理的是一个patient的数据
    Examples:
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            "incomplete":  [[False, False, False]]
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [['1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373']],
            'drugs':['1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
            'conditions_raw': ['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']
            'procedures_raw': ['4443', '4513', '3995'],
            'incomplete_raw': [False, False, False],
            'labels':['X','B','C']
        }
    """
    samples = []
    for i in range(len(patient)-1): # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        next_conditions = next_visit.get_code_list(table="DIAGNOSES_ICD")
        next_pheno = get_visit_phenotype(code_to_group, next_conditions)

        # ATC 3 level 'A04D'
        drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        filter_res = filter_lists(conditions, procedures, drugs, next_pheno)
        if filter_res is None:
            continue
        else:
            conditions, procedures, drugs, next_pheno, mask = filter_res

        # if len(conditions) * len(procedures) * len(drugs) * len(next_pheno) == 0:
        #     continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "incomplete": mask,  # [False, False, False]
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id, #['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "incomplete_raw": mask,  # [False, False, False], 被mask掉的地方为True，即缺失的地方为True
                "labels": next_pheno,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 1: # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]] 
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["incomplete"] = [samples[0]["incomplete"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]


    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["incomplete"] = samples[i - 1]["incomplete"] + [
            samples[i]["incomplete"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]

    # # remove the target drug from the history，disease prediction不需要
    # for i in range(len(samples)): # 都是最后一位
    #     samples[i]["drugs_hist"][i] = ['<pad>','<pad>','<pad>'] # 去掉target
    #     # 时序对齐的padding
    #     samples[i]["drugs_hist"] = [samples[i]["drugs_hist"][i]] + samples[i]["drugs_hist"][:i]
    return samples



def phe_prediction_miv_note_fn(patient: Patient):
    # 先经过task fn。
    samples = []
    for i in range(len(patient)-1): # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        next_conditions = next_visit.get_code_list(table="diagnoses_icd")
        next_pheno = get_visit_phenotype(code_to_group, next_conditions)

        # ATC 3 level 'A04D'
        drugs = [drug[:config['ATCLEVEL']+1] for drug in drugs]

        # exclude: visits without condition, procedure, or drug code
        filter_res = filter_lists(conditions, procedures, drugs, next_pheno)
        if filter_res is None:
            continue
        else:
            conditions, procedures, drugs, next_pheno, mask = filter_res

        # if len(conditions) * len(procedures) * len(drugs) * len(next_pheno) == 0:
        #     continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "incomplete": mask,  # [False, False, False]
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id, #['1']
                "note": extract_patient_note(int(patient.patient_id), int(visit.visit_id)), # 暂时不把它当模态, 所以有没有都行； 不然会大幅度减少数据
                "drugs": drugs,  # （used for diffusion） cur
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "incomplete_raw": mask,  # [False, False, False], 被mask掉的地方为True，即缺失的地方为True
                "labels": next_pheno,
            }
        )

    # exclude: patients with less than 1 visit
    if len(samples) < 1: # [{visit 1},{visit 2}], 有1的话，其实本身就至少有2次visit
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["incomplete"] = [samples[0]["incomplete"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]
    samples[0]["note"] = [samples[0]["note"]]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据，迭代创建
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["incomplete"] = samples[i - 1]["incomplete"] + [
            samples[i]["incomplete"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]
        samples[i]["note"] = samples[i - 1]["note"] + [
            samples[i]["note"]
        ]


    return samples


def phe_prediction_eicu_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):  # visit 次数
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        # print(visit)
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        next_conditions = next_visit.get_code_list(table="diagnosis")
        next_pheno = get_visit_phenotype(code_to_group, next_conditions)

        # exclude: visits without condition, procedure, or drug code
        filter_res = filter_lists(conditions, procedures, drugs, next_pheno)
        if filter_res is None:
            continue
        else:
            conditions, procedures, drugs, next_pheno, mask = filter_res

        # if len(conditions) * len(procedures) * len(drugs) * len(next_pheno) == 0:
        #     continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "incomplete": mask,  # [False, False, False]
                "conditions": conditions,
                "procedures": procedures,
                "drugs_hist": drugs,
                "visit_id_hist": visit.visit_id,  # ['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,
                "procedures_raw": procedures,
                "incomplete_raw": mask,
                "labels": next_pheno,
            }
        )

    if len(samples) < 1:
        return []

    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["incomplete"] = [samples[0]["incomplete"]]
    samples[0]["visit_id_hist"] = [samples[0]["visit_id_hist"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]
        samples[i]["incomplete"] = samples[i - 1]["incomplete"] + [
            samples[i]["incomplete"]
        ]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + [
            samples[i]["visit_id_hist"]
        ]
    return samples



######### los experiment


def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9



def los_prediction_miii_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # # exclude: visits without condition, procedure, or drug code
        # if len(conditions) * len(procedures) * len(drugs) == 0: # bixu药在los之前
        #     continue


        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # exclude: visits without condition, procedure, or drug code
        filter_res = filter_lists(conditions, procedures, drugs, los_category)
        if filter_res is None:
            continue
        else:
            conditions, procedures, drugs, los_category, mask = filter_res


        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "incomplete": [mask],  # [False, False, False]
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "visit_id_hist": [visit.visit_id], #['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions, # 可能出现问题所有框起来？
                "procedures_raw": procedures,
                "incomplete_raw": mask,  # [False, False, False], 被mask掉的地方为True，即缺失的地方为True
                "labels": los_category,
            }
        )

    if len(samples) < 1: # [{},{}]； 这里why
        return samples

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["incomplete"] = samples[0]["incomplete"] # 这里其实都不用
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]

        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]

        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["incomplete"] = samples[i - 1]["incomplete"] + samples[i]["incomplete"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples


def los_prediction_miv_note_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.
    """
    samples = []
    for i in range(len(patient)): # visit 次数
        visit: Visit = patient[i]

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # if len(conditions) * len(procedures) * len(drugs) == 0: # for hyperrec， 必须要放在之前。
        #     continue

        # exclude: visits without condition, procedure, or drug code
        # TODO: should also exclude visit with age < 18
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # exclude: visits without condition, procedure, or drug code
        filter_res = filter_lists(conditions, procedures, drugs, los_category)
        if filter_res is None:
            continue
        else:
            conditions, procedures, drugs, los_category, mask = filter_res


        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "incomplete": [mask],  # [False, False, False]
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "note": [extract_patient_note(int(patient.patient_id), int(visit.visit_id))],  # 暂时不把它当模态
                "visit_id_hist": [visit.visit_id], #['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions, # 可能出现问题所有框起来？
                "procedures_raw": procedures,
                "incomplete_raw": mask,  # [False, False, False], 被mask掉的地方为True，即缺失的地方为True
                "labels": los_category,
            }
        )

    if len(samples) < 2: # [{},{}]； 这里why
        return samples

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]
    samples[0]["incomplete"] = samples[0]["incomplete"] # 这里其实都不用
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]
    samples[0]["note"] = samples[0]["note"]

    for i in range(1, len(samples)): # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]

        samples[i]["incomplete"] = samples[i - 1]["incomplete"] + samples[i]["incomplete"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]
        samples[i]["note"] = samples[i - 1]["note"] + samples[i]["note"]
    
    return samples

def los_prediction_eicu_fn(patient: Patient):
    # 不知道length of stay是否有
    samples = []
    for visit in patient:
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code

        # if len(conditions) * len(procedures) * len(drugs) == 0:
        #     continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # exclude: visits without condition, procedure, or drug code
        filter_res = filter_lists(conditions, procedures, drugs, los_category)
        if filter_res is None:
            continue
        else:
            conditions, procedures, drugs, los_category, mask = filter_res


        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "incomplete": [mask],  # [False, False, False]
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs_hist": [drugs],
                "visit_id_hist": [visit.visit_id],  # ['1']
                "drugs": drugs,  # used for diffusion
                "conditions_raw": conditions,  # 可能出现问题所有框起来？
                "procedures_raw": procedures,
                "incomplete_raw": mask,  # [False, False, False], 被mask掉的地方为True，即缺失的地方为True
                "labels": los_category,
            }
        )

    if len(samples) < 1:  # [{},{}]； 这里why
        return samples

    samples[0]["conditions"] = samples[0]["conditions"]
    samples[0]["procedures"] = samples[0]["procedures"]
    samples[0]["drugs_hist"] = samples[0]["drugs_hist"]

    samples[0]["incomplete"] = samples[0]["incomplete"]  # 这里其实都不用
    samples[0]["visit_id_hist"] = samples[0]["visit_id_hist"]

    for i in range(1, len(samples)):  # 第二次，到第N次，一个patient创建一个samples数据,这个samples是遍历很多次的数据
        samples[i]["conditions"] = samples[i - 1]["conditions"] + samples[i]["conditions"]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + samples[i]["procedures"]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + samples[i]["drugs_hist"]
        samples[i]["incomplete"] = samples[i - 1]["incomplete"] + samples[i]["incomplete"]
        samples[i]["visit_id_hist"] = samples[i - 1]["visit_id_hist"] + samples[i]["visit_id_hist"]

    return samples




########## others except for task definitions
def generate_patient_group(samples, path):
    last_visits = get_last_visit_sample(samples).values()
    group_patient = {}
    for record in last_visits:
        patient_id = record['patient_id']
        incomplete = np.array(record['incomplete'], dtype=int)
        max_num = len(incomplete) * 3
        if int(max_num * 1/2) <incomplete.sum()<= int(max_num * 2/3): # 2种以上缺失。
            group_patient[patient_id] = 'G1'
        elif int(max_num * 1/3) <incomplete.sum()<=int(max_num * 1/2):
            group_patient[patient_id] = 'G2'
        elif int(max_num * 1/6) <incomplete.sum()<=int(max_num * 1/3):
            group_patient[patient_id] = 'G3'
        elif incomplete.sum()<=int(max_num * 1/6): # 一种集没有缺失。
            group_patient[patient_id] = 'G4' # 没有缺失基本

    def unique_values(dictionary):
        return set(dictionary.values())
    save_pickle(group_patient, path + 'group_patient.pkl')
    print("group patient id generate done!")
    return

# def generate_patient_group(samples, path): # OMOP数据集只有一个表征，所以没有缺失多个的。
#     """生成患者组"""
#     last_visits = get_last_visit_sample(samples).values()
#     group_patient = {'comp':[], 'miss': [], 'miss-cond':[], 'miss-proc':[], 'miss-drug':[], 'miss-more':[]}
#
#     for record in last_visits:
#         patient_id = record['patient_id']
#         incomplete = np.array(record['incomplete']) # [[1,1,0],[1,0,1],[1,1,1]]
#
#         max_num = len(incomplete) * 3
#         count = 0
#         if incomplete.sum() < max_num:
#             group_patient['miss'].append(patient_id)
#             if incomplete[:, 0].sum() < len(incomplete):
#                 group_patient['miss-cond'].append(patient_id)
#                 count += 1
#             elif incomplete[:, 1].sum() < len(incomplete) :
#                 group_patient['miss-proc'].append(patient_id)
#                 count += 1
#             elif incomplete[:, 2].sum() < len(incomplete):
#                 group_patient['miss-drug'].append(patient_id)
#                 count += 1
#             if count > 1:
#                 group_patient['miss-more'].append(patient_id)
#         else:
#             group_patient['comp'].append(patient_id)
#
#
#
#     save_pickle(group_patient, path + 'group_patient.pkl')
#     print("group patient id generate done!")
#     return


def decompress_csv_gz(folder_path):
    # 获取文件夹下所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv.gz'):
            gz_file_path = os.path.join(folder_path, filename)
            csv_file_path = os.path.join(folder_path, filename[:-3])  # 去掉.gz后缀

            # 解压文件
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(csv_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f'Decompressed: {filename} to {filename[:-3]}')

if __name__ == '__main__':
    # # create_label_for_phenotyping(config['data']['path'])
    # path = '/home/czhaobo/MMHealth/data/'
    # code_to_group, group_to_id, id_to_group = create_label_for_phenotyping(path)
    # print('generate done!')
    # # suppose we have ['01000', '01001', '01002', '01003', '01004'] for one visit, get corresponding phenotyping label
    # diagnose_visit = ['01000', '01001', '01002', '01003', '01004']
    # cur_labels = get_visit_phenotype(code_to_group, diagnose_visit)
    # print(cur_labels)
    decompress_csv_gz('/home/czhaobo/HyperHealth/data/physionet.org/files/mimic-iv-note/2.2/note/') # 暂时不要解压



