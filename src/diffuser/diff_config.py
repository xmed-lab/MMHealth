# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : diff_config.py
# Time       ：8/11/2024 5:12 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""

class DIFFCONFIG(): # 不要有drugs
    """DRL config"""
    seed = 528
    device = 6
    warm = 10
    benchmark = False

    ckpt_root = '/home/czhaobo/MMHealth/log/difflog/ckpt/'
    sample_dir = '/home/czhaobo/MMHealth/log/difflog/sample/'
    workdir = '/home/czhaobo/MMHealth/log/difflog/work_dir/' # wandb
    output_path = '/home/czhaobo/MMHealth/log/difflog/work_dir/output.log'
    z_shape = (3, 128)
    train_val_num = 50
    dim=128
    threshod = 0.4 # jaccard


    dataset = {
        'MIII':{
            'name': 'MIII',
            'feature_keys':  ['conditions', 'procedures', 'drugs'],
            'cfg' : True,
            'p_uncond' : 0.1,
        },
        'MIV-Note': {
            'name': 'MIV-Note',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.1,
        },
        'eICU': {
            'name': 'eICU',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.1,
        }
    }
    sample = {
        'mini_batch_size': 16, # 训练的时候搞大点似乎也行； 但是eCIU不行，数量太少了，一次不够；
        'scale': 1,
        'path': '/home/czhaobo/MMHealth/log/difflog/eval_sample/',
        'sample_steps': 20, # dpm solver用更少的步
        'n_samples': 10000,
    }
    hparams = 'lr' # monitor hyper

    nnet = {
        'name':'dit',
        'visit_config':  {
            'dim': 128,
            'num_heads': 2,
            'num_ids':2,
            'mode': 'sum',
            'logits_mode':'linear'
        },
        'con_config':{
            'dim': 128,
            'mode': 'concat',
            'num_heads': 2,
            'hist_mode': 'individual', # joint
        },
        'cluster_config':{
            'dim': 128,
            'k': 10, # 5， 10， 20，100
            'proto_reg': 1e-4,
            'ssl_temp': 0.1,
            'mode': 'k-means',
        },
        # 'img_size': 32,
        'in_chans' : 4,
        # 'patch_size' :  2,
        'embed_dim' :  128,
        'depth' :  2, # 12
        'num_heads' :  4,
        'mlp_ratio' :  4,
        'qkv_bias' :  False,
        'mlp_time_embed' :  False,
        'contx_dim' : 128,
        'num_contx_token' :  9 # 可能要改, 10 for note
            }


    optimizer = {
        'name': 'adamw',
        'lr' : 5e-4, #0.0002,eicu 用5e-3
        'weight_decay' : 0,#0.03,
        'betas' : (0.9, 0.9),
    }

    lr_scheduler = {
        'name': 'customized',
        'warmup_steps':5000
    }
    
    train = {
        'n_epochs': 50, # step/ epcoh
        'batch_size': 16,
        'log_interval': 10000, # 就直接存
        'eval_epoch': 5, # 3 eoch eval一次, 比max小1个
        # 'save_interval': 1000,
        # 'n_steps': 5000 # 这里的值 wrong
    }





  


config = vars(DIFFCONFIG)
config = {k:v for k,v in config.items() if not k.startswith('__')}

