# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : diff_dataset.py
# Time       ：8/11/2024 5:50 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：通用的Dataset构建方式
"""
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from .diff_utils import get_last_visit_sample


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]




class DatasetFactory(object):
    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset) # 不需要label这种东西
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        # emb->img, 暂时不写， 或许可以换位rounding
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


class CustomDataset(Dataset):
    def __init__(self, samples, note=False):
        (self.patient_id, self.visit_id, self.samples_static,
         self.samples_history, self.sample_mask, self.sample_history_mask,
         self.mask_sample_static) = samples
        self.note = note

    def __len__(self):
        return len(self.samples_static)

    def get_samples(self):
        return self.samples_static

    def get_contexts(self, num):
        """sample for train val"""
        patient_id, visit_id = [], []
        conditions, drugs, procedures, note = [], [], [], []
        mask, mask_hist = [], []
        conditions_hist, drugs_hist, procedures_hist, note_hist = [], [], [], []
        conditions_comp, drugs_comp, procedures_comp, note_compt = [], [], [], []
        for i in range(num):
            patient_id.append(self.patient_id[i])
            visit_id.append(self.visit_id[i])
            conditions.append(self.samples_static[i][0]) # 应该放完整的
            drugs.append(self.samples_static[i][1])
            procedures.append(self.samples_static[i][2])
            note.append(self.samples_static[i][3]  if self.note else 0)
            mask.append(self.sample_mask[i][:3])
            conditions_hist.append(self.samples_history[i][0])
            drugs_hist.append(self.samples_history[i][1])
            procedures_hist.append(self.samples_history[i][2])
            note_hist.append(self.samples_history[i][3] if self.note else 0) # 这里不用担心note的index out of scope，因为在custom中搞定了。
            mask_hist.append(self.sample_history_mask[i][:3])
            conditions_comp.append(self.samples_static[i][0])
            drugs_comp.append(self.samples_static[i][1])
            procedures_comp.append(self.samples_static[i][2])
            note_compt.append(self.samples_static[i][3]  if self.note else 0)
        sample_output = {
            'patient_id': patient_id,
            'visit_id': visit_id,
            'conditions': conditions,
            'drugs': drugs,
            'procedures': procedures,
            'note': note,
            'mask': mask,
            'conditions_hist': conditions_hist,
            'drugs_hist': drugs_hist,
            'procedures_hist': procedures_hist,
            'note_hist': note_hist,
            'mask_hist': mask_hist,
            'conditions_comp': conditions_comp,
            'drugs_comp': drugs_comp,
            'procedures_comp': procedures_comp,
            'note_comp': note_compt,
        }

        return sample_output

    def __getitem__(self, index):
        data_index = {
            'patient_id': self.patient_id[index],
            'visit_id': self.visit_id[index],
            'conditions': self.samples_static[index][0],
            'drugs': self.samples_static[index][1],
            'procedures': self.samples_static[index][2],
            'note':  self.samples_static[index][3] if self.note else 0,
            'mask': self.sample_mask[index][:3], # 全部只要前3个, 不需要重建note emb
            'conditions_hist': self.samples_history[index][0],
            'drugs_hist': self.samples_history[index][1],
            'procedures_hist': self.samples_history[index][2],
            'note_hist': self.samples_history[index][3] if self.note else 0,
            'mask_hist': self.sample_history_mask[index][:3],  # 标记历史的缺失情况
            'conditions_comp': self.samples_static[index][0],  # complete view
            'drugs_comp': self.samples_static[index][1],
            'procedures_comp': self.samples_static[index][2],
            'note_comp': self.samples_static[index][3] if self.note else 0,
        }
        return index, data_index





class CFGDataset(Dataset):
    def __init__(self, samples, p_uncond, empty_token, note=False):
        (self.patient_id, self.visit_id, self.samples_static,
         self.samples_history, self.sample_mask, self.sample_history_mask,
         self.mask_sample_static) = samples
        self.p_uncond = p_uncond
        self.empty_token = empty_token
        self.note = note
        if self.note:
            self.empty_mask = [0,0,0] # 可以为4， 对应修改
        else:
            self.empty_mask = [0,0,0]

    def __len__(self):
        return len(self.samples_static)

    def get_samples(self):
        return self.samples_static

    def get_contexts(self, num):
        """sample for train test"""
        patient_id, visit_id = [], []
        conditions, drugs, procedures, note = [], [], [], []
        mask, mask_hist = [], []
        conditions_hist, drugs_hist, procedures_hist, note_hist = [], [], [], []
        conditions_comp, drugs_comp, procedures_comp, note_comp = [], [], [], []
        for i in range(num):
            patient_id.append(self.patient_id[i])
            visit_id.append(self.visit_id[i])
            conditions.append(self.samples_static[i][0]) # 这里不能放mask sample，应该放完整的额 mask_sample_static， 因为她是后期加噪声。
            drugs.append(self.samples_static[i][1])
            procedures.append(self.samples_static[i][2])
            note.append(self.samples_static[i][3] if self.note else 0)
            mask.append(self.sample_mask[i][:3])
            conditions_hist.append(self.samples_history[i][0])
            drugs_hist.append(self.samples_history[i][1])
            procedures_hist.append(self.samples_history[i][2])
            note_hist.append(self.samples_history[i][3] if self.note else 0)
            mask_hist.append(self.sample_history_mask[i][:3])
            conditions_comp.append(self.samples_static[i][0])
            drugs_comp.append(self.samples_static[i][1])
            procedures_comp.append(self.samples_static[i][2])
            note_comp.append(self.samples_static[i][3] if self.note else 0)
        sample_output = {
            'patient_id': patient_id,
            'visit_id': visit_id,
            'conditions': conditions,
            'drugs': drugs,
            'procedures': procedures,
            'note': note,
            'mask': mask,
            'conditions_hist': conditions_hist,
            'drugs_hist': drugs_hist,
            'procedures_hist': procedures_hist,
            'note_hist': note_hist,
            'mask_hist': mask_hist,
            'conditions_comp': conditions_comp,
            'drugs_comp': drugs_comp,
            'procedures_comp': procedures_comp,
            'note_comp': note_comp,
        }

        return sample_output

    def __getitem__(self, index):
        if random.random() < self.p_uncond: # 不用使用这样的mask
            flag = True
        else:
            flag = False

        data_index = {
            'patient_id': self.patient_id[index],
            'visit_id': self.visit_id[index],
            'conditions': self.samples_static[index][0],
            'drugs': self.samples_static[index][1],
            'procedures': self.samples_static[index][2],
            'note': self.samples_static[index][3] if self.note else 0,
            'mask': self.sample_mask[index][:3], # if flag else self.empty_mask, # 不显式指出需要弥补的。这里略微不一样 随机的选择一部分
            'conditions_hist': self.samples_history[index][0],
            'drugs_hist': self.samples_history[index][1],
            'procedures_hist': self.samples_history[index][2],
            'note_hist': self.samples_history[index][3] if self.note else 0,
            'mask_hist': self.sample_history_mask[index][:3],  # 标记历史的缺失情况
            'conditions_comp': self.samples_static[index][0],  # complete view
            'drugs_comp': self.samples_static[index][1],
            'procedures_comp': self.samples_static[index][2],
            'note_comp': self.samples_static[index][3] if self.note else 0,
        }
        return index, data_index




class MIIIDataset(DatasetFactory):
    def __init__(self, samples, aug, cfg=False, p_uncond=None, **kwargs):
        # 这里的dataset本质上是不完整的数据集，空缺的地方为0
        super().__init__()
        print('Prepare MIII dataset...')
        self.empty_context = torch.randn(9, kwargs['config'][
            'dim'])  # for classifier free guidance 9= 3mask + 3 condition history + 3 prototype
        self.p_uncond = p_uncond
        if aug:
            train_samples, test_samples = samples
            print('Augment MIII dataset...')
            self.train = self.transform_aug_dataset(train_samples)
            self.test = self.transform_aug_dataset(test_samples)
        else:
            train_samples, test_samples = samples
            print('No Augment MIII dataset...')
            self.train = self.transform_dataset(train_samples, cfg)
            self.test = self.transform_dataset(test_samples)
        print('Prepare MIII dataset done!')
        self.mask_contexts, self.his_contexts = None, None


    def get_contexts(self, num):
        return self.test.get_contexts(num)

    def generate_mask(self, sample_static, m=3, mask_num=1):
        """[list, list, list]"""
        sample_static = sample_static.copy()
        mask_indices = np.random.choice(m, mask_num, replace=False)
        mask = np.zeros(m, dtype=bool)
        mask[mask_indices] = True
        # 使用 NumPy 进行替换
        # sample_static = np.array(sample_static, dtype=object)

        # # 转回列表（如有需要）
        # print('A',sample_static)
        # sample_static[np.array(mask)] = [['<pad>']]
        # print('B',sample_static)
        for i in range(m):
            if mask[i]:
                sample_static[i] = ['<pad>']  # 替换整个列表

        return mask.tolist(), sample_static # [list, list, ['<pad>']]


    def transform_aug_dataset(self, dataset):
        """aug: for healthcare
         19627 155310 [['51881', '5070', '2762'], ['B05X', 'A02B', 'B05B'], ['9671', '9604']] [[['51881', '5070', '2762']], [['B05X', 'A02B', 'B05B', 'C05B']], [['9671', '9604']]] [0, 0, 0] [[0, 0, 0]] [['51881', '5070', '2762', '9951'], ['B05X', 'A02B', 'B05B'], ['9671', '9604']]
        """

        # loader = samples # [{patint_id: , visit_id: , conditions: , drugs_hist: , procedures: , incomplete: }]
        patient_id = []
        visit_id = []
        samples_static = []
        mask_sample_static = [] # mask后的数据
        samples_history = []
        sample_mask = []
        sample_history_mask = []

        # # 检查是否为嵌套列表
        # sample = dataset[0]
        # if isinstance(sample['conditions_raw'][0], list):
        #     flag = True
        # else:
        #     flag = False

        for data in dataset:
            # print("CCCCCC", data)
            # break
            patient_id.append(data['patient_id'])
            visit_id.append(data['visit_id'])
            visit_static = [data['conditions_raw'], data['drugs'], data['procedures_raw']] 
            # visit_static = list(map(list, visit_static))
            samples_static.append(visit_static)
            mask_visit = visit_static.copy()
            mask_sample_static.append(mask_visit)
            sample_mask.append(data['incomplete_raw'])
            # visit_history = list(zip(data['conditions'], data['drugs_hist'], data['procedures']))
            # visit_history = list(map(list, visit_history))
            visit_length = len(data['conditions'])
            if visit_length < 2: # 这里似乎也可以修改为一样的pad即可，不然太浪费数据了。
                visit_his_cond, visit_his_drug, visit_his_proc = [['<pad>']],[['<pad>']],[['<pad>']]
                visit_his_cond_m, visit_his_drug_m, visit_his_proc_m = [True], [True], [True]
                samples_history.append([visit_his_cond, visit_his_drug, visit_his_proc])
                sample_history_mask.append([visit_his_cond_m, visit_his_drug_m, visit_his_proc_m])
            else:
                visit_history = [data['conditions'][:visit_length], data['drugs_hist'][:visit_length], data['procedures'][:visit_length]]
                samples_history.append(visit_history)
                sample_history_mask.append(data['incomplete'][:visit_length]) # 这样方便代码书写,包含自身

        recon_samples = [patient_id, visit_id, samples_static, samples_history, sample_mask, sample_history_mask, mask_sample_static]
        # print("AAAA", patient_id[0], visit_id[0], samples_static[0], samples_history[0], sample_mask[0], sample_history_mask[0], mask_sample_static[0])
        dataset = CustomDataset(recon_samples, note=False)
        return dataset


    def transform_dataset(self, samples, cfg=False):
        """train:diff_dataset

        AAAA 19627 155310 [['51881', '5070', '2762'], ['B05X', 'A02B', 'B05B'], ['<pad>']] [[['<pad>']], [['<pad>']], [['<pad>']]] [False, False, True] [[True], [True], [True]] [['51881', '5070', '2762'], ['B05X', 'A02B', 'B05B',], ['<pad>']]

        """
        # 时序转换为静态 # 最后training SAMPLE可以额外存储hard / soft emb
        samples = get_last_visit_sample(samples)
        patient_id = []
        visit_id = []
        samples_static = []
        mask_sample_static = [] # mask后的数据
        samples_history = []
        sample_mask = []
        sample_history_mask = []
        for _, patient in samples.items(): # 转为visit-level
            visit_length = len(patient['conditions'])
            for visit in range(visit_length):
                if sum(patient['incomplete'][visit])==0:
                    patient_id.append(patient['patient_id'])
                    visit_id.append(patient['visit_id'])
                    visit_static = [patient['conditions'][visit], patient['drugs_hist'][visit], patient['procedures'][visit]] # [[list, list, list]]
                    samples_static.append(visit_static)
                    mask, mask_sample = self.generate_mask(visit_static)
                    sample_mask.append(mask) #[list]
                    mask_sample_static.append(mask_sample) #[[list, list, list]]
                    if visit <2:
                        visit_his_cond, visit_his_drug, visit_his_proc = [['<pad>']],[['<pad>']],[['<pad>']]
                        visit_his_cond_m, visit_his_drug_m, visit_his_proc_m = [True], [True], [True]
                        samples_history.append([visit_his_cond, visit_his_drug, visit_his_proc])
                        sample_history_mask.append([visit_his_cond_m, visit_his_drug_m, visit_his_proc_m])
                    else:
                        samples_history.append([patient['conditions'][:visit], patient['drugs_hist'][:visit], patient['procedures'][:visit]])
                        sample_history_mask.append(patient['incomplete'][:visit]) # 这样方便代码书写,包含自身
                else:
                    # 没法检验，对于那些缺失的visit
                    continue
        recon_samples = [patient_id, visit_id, samples_static, samples_history, sample_mask, sample_history_mask, mask_sample_static]
        # print("AAAA", patient_id[0], visit_id[0], samples_static[0], samples_history[0], sample_mask[0], sample_history_mask[0], mask_sample_static[0])
        #
        if cfg:
            assert self.p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={self.p_uncond}')
            dataset = CFGDataset(recon_samples, self.p_uncond, self.empty_context)
        else:
            dataset = CustomDataset(recon_samples, note=False)
        return dataset

    # pass
    @property
    def data_shape(self):
        # batch, seq_len
        return 32, 64






class MIV_NoteDataset(DatasetFactory):
    def __init__(self, samples, aug, cfg=False, p_uncond=None, **kwargs):
        # 这里的dataset本质上是不完整的数据集，空缺的地方为0
        super().__init__()
        print('Prepare MIV-Note dataset...')
        self.empty_context = torch.randn(10, kwargs['config'][
            'dim'])  # for classifier free guidance 9= 3mask + 3 condition history + 3 prototype
        self.p_uncond = p_uncond
        if aug: # aug过程，不训练
            train_samples, test_samples = samples
            print('Augment MIV-Note dataset...')
            self.train = self.transform_aug_dataset(train_samples)
            self.test = self.transform_aug_dataset(test_samples)
        else: # 用于训练diff
            train_samples, test_samples = samples
            print('No Augment MIV-Note dataset...')
            self.train = self.transform_dataset(train_samples, cfg)
            self.test = self.transform_dataset(test_samples)
        print('Prepare MIV-Note dataset done!')
        self.mask_contexts, self.his_contexts = None, None

    def mask_ratio_partial(self, k=0.33, m=3):
        # 人为创建mask ratio; 这个mask, mask掉1/3的数据，又或许可以进行partial mask(第二个idea)
        pass

    def get_contexts(self, num):
        return self.test.get_contexts(num)

    def generate_mask(self, sample_static, m=3, mask_num=1):
        """[list, list, list]"""
        sample_static = sample_static.copy()
        mask_indices = np.random.choice(m, mask_num, replace=False) # 随机生成，这里generate MASK可以自定义。使用更多的mask模式
        mask = np.zeros(m, dtype=bool)
        mask[mask_indices] = True
        # 使用 NumPy 进行替换
        # sample_static = np.array(sample_static, dtype=object)  # 转换为 NumPy 数组

        # # 转回列表（如有需要）
        # print('A',sample_static)
        # sample_static[np.array(mask)] = [['<pad>']]
        # print('B',sample_static)
        for i in range(m):
            if mask[i]:
                sample_static[i] = ['<pad>']  # 替换整个列表

        return mask.tolist(), sample_static # [list, list, ['<pad>']]


    def transform_aug_dataset(self, dataset):
        """aug: for healthcare
         19627 155310 [['51881', '5070', '2762'], ['B05X', 'A02B', 'B05B'], ['9671', '9604']] [[['51881', '5070', '2762']], [['B05X', 'A02B', 'B05B', 'C05B']], [['9671', '9604']]] [0, 0, 0] [[0, 0, 0]] [['51881', '5070', '2762', '9951'], ['B05X', 'A02B', 'B05B'], ['9671', '9604']]
        """

        # loader = samples # [{patint_id: , visit_id: , conditions: , drugs_hist: , procedures: , incomplete: }]
        patient_id = []
        visit_id = []
        samples_static = []
        mask_sample_static = [] # mask后的数据
        samples_history = []
        sample_mask = []
        sample_history_mask = []
        for data in dataset:
            # print("CCCCCC", data)
            # break
            patient_id.append(data['patient_id'])
            visit_id.append(data['visit_id'])
            # print("AAAAAAAAAAAAAAAAAA", data['conditions_raw'])
            visit_static = [data['conditions_raw'], data['drugs'], data['procedures_raw'],data['note'][-1]] # static是4个 hist也是4个，但是mask是3个
            # visit_static = list(map(list, visit_static))
            samples_static.append(visit_static)
            mask_visit = visit_static.copy()
            mask_sample_static.append(mask_visit) # 后面mask一定要bool
            sample_mask.append(data['incomplete_raw'])
            # visit_history = list(zip(data['conditions'], data['drugs_hist'], data['procedures']))
            # visit_history = list(map(list, visit_history))
            visit_length = len(data['conditions'])
            if visit_length < 2:  
                visit_his_cond, visit_his_drug, visit_his_proc, visit_his_note = [['<pad>']],[['<pad>']],[['<pad>']], [[1.]*768]
                visit_his_cond_m, visit_his_drug_m, visit_his_proc_m, visit_his_note_m = [True], [True], [True], [True]
                samples_history.append([visit_his_cond, visit_his_drug, visit_his_proc, visit_his_note])
                sample_history_mask.append([visit_his_cond_m, visit_his_drug_m, visit_his_proc_m, visit_his_note_m])
            else:
                visit_history = [data['conditions'][:visit_length], data['drugs_hist'][:visit_length], data['procedures'][:visit_length], data['note'][:visit_length]]
                samples_history.append(visit_history)
                sample_history_mask.append(data['incomplete'][:visit_length]) # 这样方便代码书写,包含自身

        recon_samples = [patient_id, visit_id, samples_static, samples_history, sample_mask, sample_history_mask, mask_sample_static]
        # print("AAAAA", patient_id[0], visit_id[0], samples_static[0], samples_history[0], sample_mask[0], sample_history_mask[0], mask_sample_static[0])
        dataset = CustomDataset(recon_samples, note=True)
        return dataset


    def transform_dataset(self, samples, cfg=False):
        """train:diff_dataset

        AAAA 19627 155310 [['51881', '5070', '2762'], ['B05X', 'A02B', 'B05B'], ['<pad>']] [[['<pad>']], [['<pad>']], [['<pad>']]] [False, False, True] [[True], [True], [True]] [['51881', '5070', '2762'], ['B05X', 'A02B', 'B05B',], ['<pad>']]

        """
        # 时序转换为静态 # 最后training SAMPLE可以额外存储hard / soft emb
        samples = get_last_visit_sample(samples)
        patient_id = []
        visit_id = []
        samples_static = []
        mask_sample_static = [] # mask后的数据
        samples_history = []
        sample_mask = []
        sample_history_mask = []
        for _, patient in samples.items(): # 转为visit-level
            visit_length = len(patient['conditions'])
            for visit in range(visit_length):
                if sum(patient['incomplete'][visit])==0:
                    patient_id.append(patient['patient_id'])
                    visit_id.append(patient['visit_id'])
                    # 不需要真的mask
                    visit_static = [patient['conditions'][visit], patient['drugs_hist'][visit], patient['procedures'][visit], patient['note'][visit]] # [[list, list, list]]
                    samples_static.append(visit_static)
                    mask, mask_sample = self.generate_mask(visit_static, m=3)
                    sample_mask.append(mask) #[list]
                    mask_sample_static.append(mask_sample) #[[list, list, list]]
                    if visit <2:
                        visit_his_cond, visit_his_drug, visit_his_proc, visit_his_note = [['<pad>']],[['<pad>']],[['<pad>']], [[1.]*768]
                        visit_his_cond_m, visit_his_drug_m, visit_his_proc_m, visit_his_note_m = [True], [True], [True], [True]
                        samples_history.append([visit_his_cond, visit_his_drug, visit_his_proc, visit_his_note])
                        sample_history_mask.append([visit_his_cond_m, visit_his_drug_m, visit_his_proc_m, visit_his_note_m])
                    else:
                        samples_history.append([patient['conditions'][:visit], patient['drugs_hist'][:visit], patient['procedures'][:visit], patient['note'][:visit]])
                        sample_history_mask.append(patient['incomplete'][:visit]) # 这样方便代码书写,包含自身
                else:
                    # 没法检验，对于那些缺失的visit
                    continue
        recon_samples = [patient_id, visit_id, samples_static, samples_history, sample_mask, sample_history_mask, mask_sample_static]
        print("AAAA", patient_id[0], visit_id[0], samples_static[0], samples_history[0], sample_mask[0], sample_history_mask[0], mask_sample_static[0])
        #
        if cfg:
            assert self.p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={self.p_uncond}')
            dataset = CFGDataset(recon_samples, self.p_uncond, self.empty_context,  note=True)
        else:
            dataset = CustomDataset(recon_samples,  note=True)
        return dataset

    # pass
    @property
    def data_shape(self):
        # batch, seq_len
        return 32, 64




def collate_fn_dict(batchs):
    indices, batch = zip(*batchs)  # 解压索引和数据
    # data_tensor = torch.tensor(data)
    # return torch.tensor(indices), data_tensor
    return torch.tensor(indices), {key: [d[key] for d in batch] for key in batch[0]}


def get_dataset(name, aug, **kwargs):
    kwargs['aug'] = aug # 是否在aug中
    cfg = kwargs['config']['dataset'][name]['cfg']
    p_uncond = kwargs['config']['dataset'][name]['p_uncond']
    if name == 'MIII':
        return MIIIDataset(cfg=cfg, p_uncond=p_uncond, **kwargs)
    elif name == 'MIV-Note':
        return MIV_NoteDataset(cfg=cfg, p_uncond=p_uncond, **kwargs) # LOS改为MIII，不然内存爆炸
    elif name == "eICU":
        return MIIIDataset(cfg=cfg, p_uncond=p_uncond, **kwargs) # 本质没有什么差别
    else:
        raise ValueError(f"Unknown dataset: {name}")


def merge_miss(dataset, new_feature, soft=True):
    if soft:
        samples = []
        feature_keys = new_feature.keys()
        print(feature_keys)
        print((len(new_feature['conditions']), len(dataset)))
        for index, data in enumerate(dataset):
            for ind , feature_name in enumerate(feature_keys):
                
                data['new_' + feature_name] = new_feature[feature_name][index]
            samples.append(data)
        return samples
    else:
        samples = []
        feature_keys = new_feature.keys()
        for index, data in enumerate(dataset):
            data['miss_feature'] = [new_feature[feature_key][index] for feature_key in feature_keys]
            samples.append(data)

        patient_miss_data = {}
        for index, data in enumerate(samples):
            visit_id, patient_id = data['visit_id'], data['patient_id']
            if patient_id not in patient_miss_data:
                patient_miss_data[patient_id] = {}
            if visit_id not in patient_miss_data[patient_id]:
                patient_miss_data[patient_id][visit_id] = data['miss_feature']

        new_samples = []
        for index, data in enumerate(samples):
            # visit_id, patient_id = data['visit_id'], data['patient_id']
            # data['miss_feature'] = patient_miss_data[patient_id][visit_id]
            patient_id = data['patient_id']
            for index, feature_key in enumerate(feature_keys):
                data['new_' + feature_key] = [patient_miss_data[patient_id][visit][index] for visit in data['visit_id_hist']]
            # data['new_conditions'] = [patient_miss_data[patient_id][visit][0] for visit in data['visit_id_hist']]
            # data['new_drugs'] = [patient_miss_data[patient_id][visit][1] for visit in data['visit_id_hist']]
            # data['new_procedures'] = [patient_miss_data[patient_id][visit][2] for visit in data['visit_id_hist']]
            new_samples.append(data)
        print("AAAAA",samples[0])
        print("BBBBB", new_samples[0])
        return new_samples

