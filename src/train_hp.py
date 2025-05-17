# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_hp.py
# Time       ：12/11/2024 5:23 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：train healthcare prediction model
"""
import copy
import torch
from trainer import Trainer
from baselines import M3Care, SMART, PRISM, CORGAN, FlexCare, Transformer, SHAPE, DEPOT, HITNet
from models import Diffrm
from models_soft import MedDiff

from baseline_utils import filter_dataset, generate_mask, collate_fn_smart, get_dataloader, reshape_dataset, collate_fn_corgan

# from baselines.ehrgan import run_gan

def run_pcf(sample_dataset, train_dataloader, test_dataloader, config, special_input=None, exp_num='', existing_model=None, mode="multilabel", aux_data=None):
    # model definition
    model = underlying_model(config,
                             sample_dataset,
                             feature_keys=config['FEATURE'],
                             label_key=config['LABEL'],
                             special_input=special_input,
                             mode=mode
                             )
    if existing_model is not None:
        if config['MODEL'] == 'SMART':
            model = existing_model # 覆盖


    if config['TASK'] in ['PHE','REC']:
        monitor = 'jaccard_samples'
        metrics = ['jaccard_samples', 'f1_samples', 'pr_auc_samples', 'roc_auc_samples', 'precision_samples', 'recall_samples']
        # if 'p_grouped' in aux_data:
        #     metrics.append('group_rec')
    elif config['TASK'] in ["DIAG"]:
        monitor =  'roc_auc_samples'
        metrics = ['topk_acc', 'topk_precision', 'pr_auc_samples', 'roc_auc_samples']
    elif config['TASK'] == "LOS":
        monitor =  'accuracy'
        metrics = ["roc_auc_weighted_ovr", "accuracy", "cohen_kappa", "f1_weighted", "mae", "rmse"]
        # if 'p_grouped' in aux_data:
        #     metrics.append('group_cls')
    elif config['TASK'] in ["MOR", 'REA']:
        monitor = 'pr_auc'
        metrics=["roc_auc", "pr_auc", "precision", "recall"]
    else:
        raise ValueError("Task not supported!")


    trainer = Trainer(
        model=model,
        metrics=metrics,  # 换指标
        device='cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu',
        output_path=config['LOGDIR'] + config['TASK'] +  '/',
        exp_name= config['DATASET'] + '-' + config['MODEL'] + '-' + exp_num,  #
    )

    print("PCF is traininig !")
    # pass
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader, # test_dataloader,
        # test_dataloader=test_dataloader, # 检查，可能有东西没保存
        epochs=config['EPOCH'],
        warm_epochs=config['WARM_EPOCH'],
        weight_decay = config['WD'],
        # steps_per_epoch=200, # 检查
        monitor= monitor, # roc_auc
        optimizer_params={"lr": config['LR']},
        max_grad_norm=0.1,
        load_best_model_at_last=True,
        aux_data=aux_data
    )
    print("Final Test")
    scores = trainer.evaluate(test_dataloader, aux_data=aux_data)
    print(scores)
    return trainer.model


# underlying model for downstream tasks
def underlying_model(config, sample_dataset, feature_keys, label_key, mode="multilabel", special_input=None):
    choice = config['MODEL']
    if choice == "Transformer":
        return Transformer(
            sample_dataset,
            feature_keys = feature_keys,
            label_key = label_key,
            mode = mode,
            # hyper related
            dropout = config['DROPOUT'],
            num_rnn_layers = config['RNN_LAYERS'],
            embedding_dim = config['DIM'],
            hidden_dim = config['HIDDEN'],
            config = config
        )
    elif choice == "SHAPE":
        return SHAPE(
            sample_dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=config['DROPOUT'],
            num_rnn_layers=config['RNN_LAYERS'],
            embedding_dim=config['DIM'],
            hidden_dim=config['HIDDEN'],
            config=config

        )
    elif choice == "ours":
        return Diffrm(
            sample_dataset,
            feature_keys=feature_keys ,
            label_key=label_key,
            mode=mode,
            # hyper related
            dropout=config['DROPOUT'],
            num_rnn_layers=config['RNN_LAYERS'],
            embedding_dim=config['DIM'],
            hidden_dim=config['HIDDEN'],
            config=config
        )
    else:
        raise ValueError("Invalid Model choice")


def run_pretrain(sample_dataset, config, special_input=None, exp_num=''):
    """"仅限于需要pretrain的model"""
    # dataset preparation
    train_dataset, test_dataset = special_input['train'], special_input['test']
    train_dataset, test_dataset = get_pretrain_dataset(train_dataset, test_dataset, config)
    train_dataloader, test_dataloader = get_pretrain_dataloader(train_dataset, test_dataset, config)
    model = get_pretrain_model(config, sample_dataset, special_input)
    print("Pretrain initialize !")

    # model definition
    metrics, monitor, monitor_criterion = get_metrics(config) # 其实大多都是重构mse,这些


    # model training
    trainer = Trainer(
        model=model,
        metrics=metrics,  # 换指标
        device='cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu',
        output_path=config['LOGDIR'] + config['TASK'] +  '/',
        exp_name= config['DATASET'] + '-' + config['MODEL'] + '-pre'+ '-' + exp_num,  # pretrain
    )

    print("Pretrain model is traininig !")

    # pass
    if config['MODEL'] == 'CORGAN':
        # for i in train_dataloader:
        #     print("XXXX", i)
        #     break
        trainer.train_auto(
            train_dataloader=train_dataloader,

            epochs=15, # 不要训练太长时间
            weight_decay = config['WD'],
            # steps_per_epoch=200, # 检查
            monitor= monitor, # roc_auc
            monitor_criterion=monitor_criterion,
            optimizer_params={"lr": config['LR']},
            max_grad_norm=0.1,
            load_best_model_at_last=True,
            ema_used=False,
        )
        print("Train autoencoder Done!")
        trainer.model.mode = 'binary'
        trainer.metrics = ['roc_auc', 'pr_auc', 'f1']
        monitor = 'roc_auc'
        trainer.train_gan(
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader, # test_dataloader,
            # test_dataloader=test_dataloader, # 检查，可能有东西没保存
            epochs=20,
            weight_decay = config['WD'],
            # steps_per_epoch=200, # 检查
            monitor= monitor, # roc_auc
            monitor_criterion=monitor_criterion,
            optimizer_params={"lr": config['LR'],  'b1': 0.9, 'b2':0.999, 'n_iter':4}, # n_iter指的是discriminator更新的次数
            max_grad_norm=0.1,
            load_best_model_at_last=True,
            ema_used=False,
        )
        print("Train GAN Done!")

        # model test
        print("Final Test")
        scores = trainer.evaluate_gan(test_dataloader, aux_data={})
        print(scores)
    elif config['MODEL'] == 'SMART':
        trainer.train( # 这里其实应该train.pretrain
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader, # test_dataloader,
            test_dataloader=test_dataloader, # 检查，可能有东西没保存
            epochs=20,
            weight_decay = config['WD'],
            # steps_per_epoch=200, # 检查
            monitor= monitor, # roc_auc
            monitor_criterion=monitor_criterion,
            optimizer_params={"lr": config['LR']},
            max_grad_norm=0.1,
            load_best_model_at_last=True,
            ema_used=True,
        )

        # model test
        print("Final Test")
        # scores = trainer.evaluate(test_dataloader, aux_data={}) # 除非使用
        # print(scores)

    # return model
    return trainer.model


def get_pretrain_model(config, sample_dataset, special_input):
    if config['MODEL'] == 'SMART':
        return SMART(
            sample_dataset,
            feature_keys=config['FEATURE'],
            label_key=config['LABEL'],
            mode="regression",
            # hyper related
            dropout=config['DROPOUT'],
            num_rnn_layers=config['RNN_LAYERS'],
            embedding_dim=config['DIM'],
            hidden_dim=config['HIDDEN'],
            config=config,
            train_mode='pretrain'
        )
    elif config['MODEL'] == 'CORGAN':
        return CORGAN(
            sample_dataset,
            feature_keys=config['FEATURE'],
            label_key=config['LABEL'],
            mode="regression",
            # hyper related
            dropout=config['DROPOUT'],
            num_rnn_layers=config['RNN_LAYERS'],
            embedding_dim=config['DIM'],
            hidden_dim=config['HIDDEN'],
            config=config
        )
    else:
        raise ValueError("Pretrain not supported!")

def get_pretrain_dataset(train_dataset, test_dataset, config):
    train_dataset, test_dataset = copy.deepcopy(train_dataset), copy.deepcopy(
        test_dataset)  # 这个非常重要，不然会修改sample_dataset
    if config['MODEL'] == 'SMART':
        # filter
        train_dataset = filter_dataset(train_dataset, config)
        test_dataset = filter_dataset(test_dataset, config)
        # generate mask
        train_dataset = generate_mask(train_dataset, config)
        test_dataset = generate_mask(test_dataset, config)
        return train_dataset, test_dataset
    elif config['MODEL'] == 'CORGAN':
        # filter
        train_dataset = filter_dataset(train_dataset, config)
        test_dataset = filter_dataset(test_dataset, config)
        # reshape
        train_dataset = reshape_dataset(train_dataset, config)
        test_dataset = reshape_dataset(test_dataset, config)
        return train_dataset, test_dataset
    else:
        pass

def get_pretrain_dataloader(train_dataset, test_dataset, config):
    if config['MODEL'] == 'SMART':
        collate_fn = collate_fn_smart
        train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True,
                                          collate_fn=collate_fn)
        test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=True,
                                         collate_fn=collate_fn)
        return train_dataloader, test_dataloader
    elif config['MODEL'] == 'CORGAN':
        collate_fn = collate_fn_corgan
        train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True,
                                          collate_fn=collate_fn)
        test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=True,
                                            collate_fn=collate_fn)
        return train_dataloader, test_dataloader
    else:
        raise ValueError("Loader not supported!")


def load_pretrain_model(sample_dataset, config, special_input=None, exp_num='', mode="multilabel"):
    if config['MODEL'] == 'SMART':
        model = underlying_model(config,
                                 sample_dataset,
                                 feature_keys=config['FEATURE'],
                                 label_key=config['LABEL'],
                                 special_input=special_input,
                                 mode=mode
                                 )
        ckpt_path = config['LOGDIR'] + config['TASK'] +  '/' + config['DATASET'] + '-' + config['MODEL'] + '-pre'+'-' + exp_num + '/best.ckpt'
        print("Pretrain Load dir, ", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location='cuda:' + config['GPU'])
        model.load_state_dict(state_dict, strict=False)
        model.train_mode ='finetune' # 防止被覆盖
    elif config['MODEL'] == 'CORGAN':
        model = underlying_model(config,
                                 sample_dataset,
                                 feature_keys=config['FEATURE'],
                                 label_key=config['LABEL'],
                                 special_input=special_input,
                                 mode=mode
                                 )
        ckpt_path = config['LOGDIR'] + config['TASK'] +  '/' + config['DATASET'] + '-' + config['MODEL'] + '-pre'+'-' + exp_num + '/best.ckpt'
        print("Pretrain Load dir, ", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location='cuda:' + config['GPU'])
        model.load_state_dict(state_dict, strict=False)
        for param in model.generator.parameters():
            param.requires_grad = False
        for param in model.autoencoder.parameters():
            param.requires_grad = False


    else:
        raise ValueError("Load not supported!")
    return model


def get_metrics(config):
    if config['MODEL'] in ['SMART', 'CORGAN']:
        metrics = ['rmse', 'mae']
        monitor = 'rmse'
        monitor_criterion = 'min'
        return metrics, monitor, monitor_criterion
    else:
        raise ValueError("Metrics not supported!")



def get_task_fn(config):
    dataset = config['DATASET']
    task = config['TASK']
    # 临时实现
    from data import phe_prediction_miii_fn, phe_prediction_miv_note_fn, phe_prediction_eicu_fn
    from data import los_prediction_miii_fn, los_prediction_miv_note_fn, los_prediction_eicu_fn # 可能要变动

    all_fn = {
        'MIII': {
            'PHE_fn': phe_prediction_miii_fn,
            'DIAG_fn': None,
            'LOS_fn': los_prediction_miii_fn,
            'MOR_fn': mor_prediction_miii_fn, # 下次补上
            'REA_fn': read_prediction_miii_fn,
            'REC_fn': rec_prediction_miii_fn,
        },
        'MIV': {
            'PHE_fn': None,
            'DIAG_fn': None,
            'LOS_fn': None,
            'MOR_fn': None,  # 下次补上
            'REA_fn': None,
            'REC_fn': rec_prediction_miv_fn,
        },
        'MIV-Note': {
            'PHE_fn': phe_prediction_miv_note_fn,
            'DIAG_fn': None,
            'LOS_fn': los_prediction_miv_note_fn,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': rec_prediction_miv_note_fn,
        },
        'eICU': {
            'PHE_fn': phe_prediction_eicu_fn,
            'DIAG_fn': None,
            'LOS_fn': los_prediction_eicu_fn,
            'MOR_fn': None,
            'REA_fn': None,
            'REC_fn': rec_prediction_eicu_fn,
        },

    }
    task_fn = all_fn[dataset][task + '_fn']
    if task_fn is None:
        raise ValueError("Task not implementation!")
    if task in ['PHE', 'DIAG', 'REC']:
        mode = 'multilabel'
    elif task in ['LOS']:
        mode = 'multiclass'
    elif task in ['MOR', 'REA']:
        mode = 'binary'
    else:
        raise ValueError("Mode not supported!")
    return task_fn, mode


