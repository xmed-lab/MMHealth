# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main_diff.py
# Time       ：8/11/2024 3:58 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：diffusion的main,编译器悬python3; https://github.com/U-ViT/blob/main/train_t2i_discrete.py
"""
import os
import tempfile
import einops
import builtins
import wandb
import accelerate
import ml_collections
from absl import logging
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from diffuser import auto
from diffuser.diff_config import config
from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from diffuser.fidscore import calculate_fid_given_paths
from torch.utils._pytree import tree_map


def run_diffusion(config):
    print("run_diffusion")
    mp.set_start_method('spawn') # 多进程
    config = ml_collections.FrozenConfigDict(config)
    
    # 加速
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision # 混合精度
    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process: # 主进程用作模型加载，记录日志等功能
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'mmht_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline') # 离线存储
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error') # 禁用打印
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    # dataset
    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat) # fid起到啥作用？

    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, drop_last=True,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    # model define
    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root) # load latest ckpt

    autoencoder = auto.get_visit_latent(**config.autoencoder)
    autoencoder.to(device)

    clust = auto.get_cluster(**config.clust)
    clust.to(device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)



    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data
    
    data_generator = get_data_generator() # 转为生成器

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                yield _context

    context_generator = get_context_generator() # 这个context是啥？condition吗？

    # diffusion prepare
    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas)
    logging.info(f'use {_schedule}')


    def cfg_nnet(x, timesteps, context):
        # classifier-free config
        _cond = nnet_ema(x, timesteps, context=context) # 果然是condition
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)


    def train_step(_batch):
        # 训练步骤，预测噪声， 这里要改成LM rounding估计。
        _metrics = dict()
        optimizer.zero_grad()
        _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
        loss = utils.LSimple(_z, nnet, _schedule, context=_batch[1])  # currently only support the extracted feature version
        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)


    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        # 逐步去噪声。这里NoiseVP和DPM-solver是固定的模版，不需要改动。，只需要把sample的东西和对应的去燥过程写好即可。
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float()) # 选一个加噪声的方式

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.) # embedding， 感觉这里需要rounding
        return decode(_z)


    def eval_step(n_samples, sample_steps):
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _context = next(context_generator)
            assert _context.size(0) == _n_samples
            return dpm_solver_sample(_n_samples, sample_steps, context=_context)

        with tempfile.TemporaryDirectory() as temp_path: # 创建临时目录，进行文件管理
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process: # 采样完成后计算fid和原图的差异。metrics,可以用于可视化
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator)) # tree_map更快
        metrics = train_step(batch)

        nnet.eval()
        # 记录metrics, 输出train_state.step, 以及config.train.log_interval
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        # 输出eval
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0: # 任何记录日志，保存模型都要使用
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            samples = dpm_solver_sample(_n_samples=2 * 5, _sample_steps=50, context=contexts)
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        # save ckpt
        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps) # 选取最好的ckpt



if __name__ == '__main__':
    config = config
    run_diffusion(config)