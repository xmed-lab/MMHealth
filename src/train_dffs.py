# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_dffs.py
# Time       ：11/11/2024 4:05 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：仿照CV重建
"""

import os
import torch
import tempfile
import einops
import wandb
import torch.nn.functional as F
import numpy as np
from absl import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from utils import set_random_seed
from utils import pad_list




def run_diffusion(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    """
    :param config: diffusion config
    :param config_health: healthcare config
    :param hp_dataset: healthcare train dataset
    :param tokenizers: healthcare tokenizers
    :return:
    """
    print("run_diffusion")
    set_random_seed(config_health['SEED'])

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # utils.calculate_quality(config['sample']['path'] + data_n)

    # 加速
    # accelerator = accelerate.Accelerator() # CUDA_VISIBLE_DEVICES=4  accelerate launch main.py
    # device = accelerator.device
    # accelerate.utils.set_seed(config['seed'], device_specific=True)
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')

    # logging.info(f'Process {accelerator.process_index} using device: {device}')
    logging.info(f'using device: {device}')

    # config['mixed_precision'] = accelerator.mixed_precision  # 混合精度
    # assert config['train']['batch_size'] % accelerator.num_processes == 0
    batch_size = config['train']['batch_size'] #// accelerator.num_processes

    # if accelerator.is_main_process:  # 主进程用作模型加载，记录日志等功能
    os.makedirs(config['ckpt_root'] + data_n + '/' + task, exist_ok=True)
    os.makedirs(config['sample_dir'] + data_n + '/' + task, exist_ok=True)

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    wandb.init(dir=os.path.abspath(config['workdir']), project=f'mmht_{data_n}_{task}', config=config,
               name=config['hparams'], job_type='train', mode='offline')  # 离线存储
    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n +'-'+ task +'-' + exp_num + '-output.log'))
    logging.info(config)
    # else:
    #     utils.set_logger(log_level='error')  # 禁用打印
    #     builtins.print = lambda *args: None
    # logging.info(f'Run on {accelerator.num_processes} devices')

    # dataset
    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config) # training 所以aug设置为False
    # assert os.path.exists(dataset.fid_stat)  # fid起到啥作用？label的作用

    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)

    print("Input for diffusion", train_dataset[0])
    print("AAAAAAAA", (len(train_dataset)//batch_size +1)*batch_size, len(train_dataset)) # 如果dropout为False，就不要+1

    # model define
    train_state = utils.initialize_train_state(config, device, tokenizers, len(train_dataset)) # 多一个batch
    # nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config['ckpt_root'] + data_n + '/' + task)  # load latest ckpt


    # @ torch.cuda.amp.autocast() # 半精度

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:], feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B
            if feature_key == 'conditions':
                truncate_indices = np.argsort(array_prob, axis=1)[:, -20:]+2 # depends on dataset
                tokens = tokenizers[feature_key].batch_decode_2d(truncate_indices.tolist()) # indices+2
            else:
                array_prob[array_prob >= thred] = 1
                array_prob[array_prob < thred] = 0  # 优化同步
                indices = []
                for index, row in enumerate(array_prob):
                    indice = (np.where(row == 1)[0]+2).tolist()
                    if len(indice) <1: # 如果没有预测到，就取最大的作为预测
                        indice = [max_indices[index]+2] # 某一行最大的那个
                    indices.append(indice)
            # indices = [(np.where(row == 1)[0]+2).tolist() for row in array_prob] # 加上padding项
                tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens

        return _batch, hard_samples

    def get_data_generator():
        while True:
            for indices, data in tqdm(train_dataset_loader, desc='epoch'): # disable=not accelerator.is_main_process
                yield indices, data

    data_generator = get_data_generator()  # 转为生成器

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()

    # diffusion prepare
    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas) # forward add noise schedule
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context):
        # classifier-free config
        _cond = nnet_ema(x, timesteps, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond #+ config['sample']['scale'] * (_cond - _uncond) # classifier free 要用unconditional的表征进行放缩。

    def train_step(indices, _batch, epoch=None):
        _metrics = dict()
        optimizer.zero_grad()
        # _z, _z_mask, _z_his = encode(_batch[0], _batch[1], batch[2]) # B, 3, D; B, 3, D
        # _zc, loss_c = cluster(_z) # B, 1, D, near cluster

        _z, _context, neigh_loss, _, _ = nnet.encode_everything(indices, **_batch) # B，3，D； B,9,D； 对比损失

        loss_mse, loss_round, T_loss = utils.LSimple(_z, nnet, _schedule,
                             context=_context,  **_batch)  # currently only support the extracted feature version
        _metrics['loss_mse'] = loss_mse.detach().mean() #accelerator.gather(loss.detach()).mean()
        _metrics['loss_c'] = neigh_loss.detach().mean() #accelerator.gather(neigh_loss.detach()).mean()
        _metrics['loss_rounding'] = loss_round.detach().mean() # batch
        _metrics['T_loss'] = T_loss.detach().mean() # batch
        total_loss = loss_mse.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean() if epoch > config['warm'] else loss_mse.mean() + loss_round.mean() + T_loss.mean()
        total_loss.backward()
        # accelerator.backward(loss.mean() + neigh_loss.mean())

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return _z, dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device) # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1,1, config['z_shape'][1]) # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2) # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t ,kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)  # embedding， 感觉这里需要rounding
        return decode(_z, nnet.get_logits)


    def eval_step(n_samples, sample_steps):
        """这里需要做case study"""
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config["sample"]["mini_batch_size"]}')

        def sample_fn(_n_samples):
            indices, batch = next(test_generator)

            assert indices.size(0) == _n_samples
            indices, sample_input = convert_input((indices, batch))
            _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
            # _context_emb = encode(_contexts)
            return labels, hard_labels, dpm_solver_sample(_n_samples, sample_steps, context=_context_emb, z_init=_z, mask=sample_input['mask'])

        with tempfile.TemporaryDirectory() as temp_path:  # 创建临时目录，进行文件管理
            path = config['sample']['path'] + data_n + '/' + task or temp_path
            work_dir = config['workdir'] + data_n + '/' + task
            # if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)
            utils.sample2dir(path, n_samples, config['sample']['mini_batch_size'], sample_fn, tokenizers,
                             dataset.unpreprocess) # 看看要不要改成test，因为这里是test

            _quality = 0
            # if accelerator.is_main_process:  # 采样完成后计算fid和原图的差异。metrics,可以用于可视化
            # _fid = None
            _quality, jaccard_quality = utils.calculate_quality(path)
            logging.info(f'step={train_state.step} quality-{n_samples}={_quality}')
            logging.info(f'step={train_state.step} jaccard_quality-{n_samples}={jaccard_quality}')
            with open(os.path.join(config['workdir'] + data_n + '/' + task, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} quality-{n_samples}={_quality} jaccard_quality-{n_samples}={jaccard_quality}', file=f)
            wandb.log({f'quality_mse-{n_samples}': _quality}, step=train_state.step)
            _quality = torch.tensor(_quality, device=device)
            # _fid = accelerator.reduce(_fid, reduction='sum')

        return _quality.item(), jaccard_quality # 这里存储的是embeddings，一旦不进行decode就是mediffusion。

    # logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config["mixed_precision"]}')

    logging.info(f'Start fitting, step={train_state.step}')

    def convert_input(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data
        # print("batch_data 0", batch['mask'])

        feature_keys = ['conditions', 'drugs', 'procedures']
        # batch_new = {}
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key]) # B,M,  m——feature没有用。
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist']) # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp']) # B,M
            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
            batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
            # mask
            batch[feature_key + '_mask'] = batch[feature_key] != 0 # 为True的就是mask掉的, # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0 # 这个是为了padding, # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0 # 这个是为了padding # B,M

            # others
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,  truncation=False) # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
            batch[feature_key + '_comps_origin'] = origin # origin ID

            # note special
            batch['has_note'] = False
            if config_health['DATASET'] == 'MIV-Note': # LOS MIV修改下。
                batch['note'] = torch.tensor(batch['note'], device=device) # cur, B,D; 这里不需要对他进行重建，因为他可能变为pad字符串
                batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)# hist, B,T,D
                batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) !=0 # B,T

                # 作为context,需要合并
                batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
                batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,
                batch['note' + '_hist_mask'] = torch.cat(
                    [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1

                batch['has_note'] = True

        # mask设置为固定
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        # mask不固定, evaldffs固定就好, diverse
        else:
            m = 3 # 4 if batch['has_note'] else 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0],m=m).to(device=device)


        aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in feature_keys] # B,V,1
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3，记录history mask状态

        return indices, batch


    step_quality = []
    jaccard_quality = []
    # while train_state.step < config['train']['n_steps']:
    max_n_steps = (len(train_dataset) // config['train']['batch_size']) + 1 if len(train_dataset) % config['train']['batch_size'] !=0 else len(train_dataset) // config['train']['batch_size'] # 不对，如果正好整除的话，就不需要+1
    for epoch in range(config['train']['n_epochs']):
        if epoch == 0:
            nnet.e_step()  # 初始化
        # print("AAAAAAAAA", nnet.total_visit_emb.shape, max_n_steps, len(train_dataset))
        total_visit_embs = []
        total_indices = []
        for i in range(max_n_steps):
            nnet.train()
            # batch = tree_map(lambda x: convert_input(x), next(data_generator))  # tree_map更快
            indices, batch = convert_input(next(data_generator), mask_static=False)

            _z, metrics = train_step(indices, batch, epoch)

            total_visit_embs.append(_z.view(_z.shape[0],-1)) # B, 3D
            total_indices.append(indices) # B,1

            nnet.eval()
            # 记录metrics, 输出train_state.step, 以及config.train.log_interval
            # if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            if train_state.step % config['train']['log_interval'] == 0:
                logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                logging.info(config['workdir'])
                wandb.log(metrics, step=train_state.step) # 不同step

            # 中途输出
            # if train_state.step % config['train']['eval_interval'] == 0:  # 任何记录日志，保存模型都要使用
            #     torch.cuda.empty_cache()
                logging.info('Save a grid of {} samples for training...'.format(config['train_val_num']))
                sample_input = dataset.get_contexts(num=config['train_val_num'])
                indices = torch.arange(config['train_val_num']).to(device)
                indices, sample_input = convert_input((indices, sample_input))
                _z, contexts, _, labels, hard_labels = nnet.encode_everything_test(indices, **sample_input)

                # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5] # 选取前10个context
                samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50, context=contexts, z_init=_z, mask=sample_input['mask'])
                # print("generate", hard_samples['conditions'][0])
                # print("origin", tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])[0]) # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['procedures'][0])
                # print("origin", tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])[0])  # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['drugs'][0])
                # print("origin", tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])[0])  # 没啥问题，encode, decode用同一套
                # samples = make_grid(dataset.unpreprocess(samples), 5)
                # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
                jaccard_condition = utils.calculate_average_jaccard(hard_samples['conditions'], tokenizers['conditions'].batch_decode_2d(hard_labels['conditions']))
                jaccard_procedure = utils.calculate_average_jaccard(hard_samples['procedures'], tokenizers['procedures'].batch_decode_2d(hard_labels['procedures']))
                jaccard_drug = utils.calculate_average_jaccard(hard_samples['drugs'], tokenizers['drugs'].batch_decode_2d(hard_labels['drugs']))
                mse_loss = F.mse_loss(samples, labels.squeeze())
                logging.info({'mask num': sample_input['mask'].sum(dim=0), 'val samples mse': mse_loss, 'jaccard': (jaccard_condition,jaccard_procedure,jaccard_drug)})
                # wandb.log({'samples': mse_loss }, step=train_state.step)
                # torch.cuda.empty_cache()
            # accelerator.wait_for_everyone()

            # save ckpt， 这一块代码检查当前进程是否为主进程（accelerator.is_main_process），如果是则保存模型
        # if train_state.step % config['train']['save_interval'] == 0 or train_state.step == config['train']['n_steps']:
        # torch.cuda.empty_cache()
        logging.info(f'Save and eval checkpoint {epoch}...')
        # if accelerator.local_process_index == 0:
        train_state.save(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch}.ckpt'))
        # accelerator.wait_for_everyone()
        if epoch==0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0):
            quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=50)  # calculate fid of the saved checkpoint
        else:
            quality = 10000 # 因为靠mse取最小的mse
            jaccard_tuple = (1,1,1)
        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))
        # torch.cuda.empty_cache()

        # e step
        total_visit_embs = torch.cat(total_visit_embs, dim=0)
        total_indices = torch.cat(total_indices, dim=0)
        total_visit_embs = total_visit_embs[total_indices.argsort()] # 重新排序， 这里需要check下对不对。
        print("CCCCCCCC", total_visit_embs.shape)
        nnet.update_train_visit_emb(total_visit_embs)
        visit_centroids, _ = nnet.e_step()
        nnet.update_centroid_emb(visit_centroids)



    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'step_quality: {jaccard_quality}')
    epoch_best = sorted(step_quality, key=lambda x: x[1])[0][0] # 最小那个mse
    logging.info(f'step_best: {epoch_best}')
    train_state.load(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt'))
    del metrics
    # accelerator.wait_for_everyone()
    eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])  # 选取最好的ckpt

    nnet_path = os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt/nnet.pth')
    return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]

