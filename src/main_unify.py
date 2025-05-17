# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：23/10/2024 4:55 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：# 为HyperHealth, diag
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import time
import torch

from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, eICUDataset, OMOPDataset # SampleEHRDataset
from pyhealth.datasets import SampleEHRDataset as SampleEHRDatasets
# from data import SampleEHRDatasetSIMPLE as SampleEHRDataset #
from pic_parse import PICDataset
from omix_parse import OMIXDataset
from utils import split_by_patient, load_pickle, save_pickle, set_random_seed, get_tokenizers
from loader import get_dataloader, get_special_input
from data import convert_dataset
from diffuser.diff_config import config as diff_config
from config import config
from train_dffs import run_diffusion
from eval_dffs import augment_dataset, load_diffusion, evaluate_diff
from train_hp import run_pcf, run_pretrain, load_pretrain_model, get_task_fn
from baseline_utils import missing_statistics
from data import generate_patient_group
set_random_seed(config['SEED'])
from shot_parse import SHOTDataset

class SampleEHRDataset(SampleEHRDatasets):
    # 在制造数据时候可以避免繁琐的检查,
    def _validate(self):
        return True

def get_warm_cold_split(samples):
    from pyhealth.datasets import split_by_patient
    sample_dataset = SampleEHRDatasets(samples)
    _,_,warm_test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=1.0,  # Train test split
        warm_cold='warm',
        seed=config['SEED']
    )
    _, _, cold_test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=1.0,  # Train test split
        warm_cold='cold',
        seed=config['SEED']
    )
    print("Success split warm - cold !")
    return warm_test_dataset, cold_test_dataset





def run_single_config(augment=False, retrain=False, exp_num='3', pretrain=False):
    """
    :param augment: whether augmented dataset
    :param retrain: whether retrain the diffusion model
    :param exp_num:
    :return:
    """
    # load datasets
    # STEP 1: load data
    root_to = '/home/czhaobo/MMHealth/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])

    task_fn, mode = get_task_fn(config)
    if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
        print("Prepare dataset!")
        if config['DATASET'] == 'MIII':
            base_dataset = MIMIC3Dataset(
                root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciii/1.4",
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 这里graphcare的ATC-level是3；和我们在data阶段有差别
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'eICU':
            base_dataset = eICUDataset(
                root="/home/czhaobo/HyperHealth/data/physionet.org/files/eicu-crd/2.0",
                tables=["diagnosis", "medication", "physicalExam", "treatment", "admissionDx"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV':
            base_dataset = MIMIC4Dataset(
                root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV-Note': # 这里可以通过subject_id进行联系。因为有一部分数据被MIMIC官方雪藏了。所有用2.2以上的版本
            base_dataset = MIMIC4Dataset(
                root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False,
                refresh_cache=False,
            )

        else:
            print("No such dataset!")
            return
        base_dataset.stat()
        # set task
        sample_dataset = base_dataset.set_task(task_fn)
        sample_dataset.stat()
        samples = sample_dataset.samples
        save_pickle(samples, root_to + 'datasets_pre_stand.pkl')


        print("initial dataset done!")
        print("Please run again!")
        return
    else:
        start = time.time()
        samples = load_pickle(root_to + 'datasets_pre_stand.pkl')
        # load_code_convert(dataset=config['DATASET'], samples=samples) # 如果这里要弄，必须要重新生成rare disease

        # group
        generate_patient_group(samples, root_to)
        p_grouped = load_pickle(root_to + 'group_patient.pkl')

        end = time.time()
        print("Load data done! Cost time {} s".format(end-start))

        if config['DEV']:
            print("DEV train mode: 1000 patient")
            samples = samples[:3000]
            sample_dataset = SampleEHRDataset(# 这个贼耗时
                samples,
                dataset_name=config['DATASET'],
                task_name=config['TASK'],
            )
            train_dataset, val_dataset, test_dataset = split_by_patient(
                sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                train_ratio=1.0,  # Train test split
                seed=528
            )
            del samples

            endt = time.time()
            print('Dataset done!, Cost {} s'.format(endt - end))
        else:
            try:
                reserved_tensor = torch.ones((10, 10)).to('cuda:' + config['GPU']) # 占据GPU
                print("GPU Memory Usage", torch.cuda.memory_allocated('cuda:' + config['GPU']) / 1024 / 1024 / 1024,
                      "GB")
                if os.path.exists(root_to + 'train_dataset.pkl'):
                    sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False, all=True) # 需要查看是否需要load_code_convert
                    # warm_test, cold_test = get_warm_cold_split(samples)
                    train_samples = load_pickle(root_to + 'train_dataset.pkl')
                    val_samples = load_pickle(root_to + 'val_dataset.pkl')
                    test_samples = load_pickle(root_to + 'test_dataset.pkl')
                    train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                    val_dataset = convert_dataset(val_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                    test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                else:
                    sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=True)

                    train_dataset, val_dataset, test_dataset = split_by_patient(
                        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                        train_ratio=1.0,  # Train test split
                        warm_cold=False, # 所有的warm一次性看完。
                        seed=config['SEED']
                    )  # 这样似乎更快，固定随机种子的时候是一样的；
                    # train_samples, _, test_samples = achieve_samples(train_dataset), _, achieve_samples(test_dataset)
                    save_pickle(train_dataset, root_to + 'train_dataset.pkl')
                    save_pickle(val_dataset, root_to + 'val_dataset.pkl')
                    save_pickle(test_dataset, root_to + 'test_dataset.pkl')
            finally:
                del reserved_tensor
                # torch.cuda.empty_cache() # 会突然爆oom

            endt = time.time()
            print('Train Dataset done!, Cost {} s'.format(endt - end))


    # STEP 2: load dataloader
    if augment:
        diff_config['device'] = int(config['GPU'])
        if config['DATASET'] == "MIV-Note": # 要改train_diff, eval_diff
            diff_config['nnet']['num_contx_token'] = 10
        tokenizers = get_tokenizers(sample_dataset)
        if retrain or not os.path.exists(diff_config['ckpt_root']+ config['DATASET']):
            print('start retrain diffusion model!')
            print('Sample data of 1 patient, input for diffusion dataset ', train_dataset[0])
            best_model_path, generated_model, num_train_visit = run_diffusion(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏
            with open(diff_config['ckpt_root'] + config['DATASET'] + '/' + config['TASK'] + "/best_model_path.txt", "w") as file:
                print(best_model_path, file=file)
                print(num_train_visit, file=file)

            diff_config["nnet_path"] = best_model_path
            diff_config["num_train_visit"] = num_train_visit

            train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
            print('retrain diffusion model done!')
        else:
            if not os.path.exists(root_to + 'train_samples.pkl'):
                with open(diff_config['ckpt_root'] + config['DATASET'] + '/' + config['TASK']+ "/best_model_path.txt", "r") as file:
                    lines = file.readlines()  # 逐行读取内容
                # 获取内容
                best_model_path = lines[0].strip()  # 第一个内容
                num_train_visit = int(lines[1].strip())  # 第二个内容，转换为整数

                # 输出读取的内容
                print("Best model path:", best_model_path)
                print("Number of training visits:", num_train_visit)

                diff_config["nnet_path"] = best_model_path
                diff_config["num_train_visit"] = num_train_visit

                print('Load diffusion model!')
                generated_model = load_diffusion(diff_config, tokenizers)

                print('Load diffusion model done!')
                train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
            else:
                print("Have been augmented!")
                del val_dataset, test_dataset
                train_samples = load_pickle(root_to + 'train_samples.pkl')
                test_samples = load_pickle(root_to + 'test_samples.pkl')

        train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        # print("AAAAAA", train_dataset[0])
        print("Augment Done!")

    collate_fn = get_special_input(config)
    # missing_statistics(train_dataset, config)
    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn)
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn) # 少数类可能抽不到，草
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))

    # cache clear
    # torch.cuda.empty_cache() # 占用内存清理
    gc.collect()


    print("===============Train Healthcare Prediction Model!===============")
    existing_model = None
    if pretrain:
        print("Pretrain on the baseline models")
        start = time.time()
        run_pretrain(sample_dataset, config, exp_num=exp_num, special_input={'train': train_dataset, 'test': test_dataset})
        end_time = time.time()

        model = load_pretrain_model(sample_dataset, config, special_input=None, exp_num=exp_num, mode=mode)
        print("Pretrain done! Cost {} s".format(end_time - start))
        existing_model = model

    start = time.time()
    run_pcf(sample_dataset, train_dataloader, test_dataloader, config, exp_num=exp_num, existing_model=existing_model, mode=mode, aux_data={'p_grouped': p_grouped})
    end = time.time()
    print("Training done! Cost {} s".format(end - start))






if __name__ == '__main__':
    retrain = False
    augment = True # ours要开
    pretrain = False # 只有在一些二次训练的baseline上面开启。SMART, CORGAN
    if config['MODEL'] in  ['ours', 'MedDiff']:
        retrain = False
        augment = True
    elif config['MODEL'] == 'SMART' or config['MODEL'] == 'CORGAN':
        pretrain = True
    else:
        print('No need to retrain or augment!')

    exp_num = '20' # 1 为aba；20 为rebutall
    print("Hi, This is Diffrm Health!")
    print("You are running on", config['DATASET'], "dataset! ", config['TASK'], "task!", config['MODEL'], "model!")
    run_single_config(augment=augment, retrain=retrain, exp_num=exp_num, pretrain=pretrain)
    print("All Done!")
    # du -sh ~/* | sort -h ; disk

