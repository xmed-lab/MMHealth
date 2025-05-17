# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : eval_diff.py
# Time       ：11/11/2024 11:30 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：根据训练好的模型进行aug
"""
import os
import builtins
import einops
import ml_collections
import torch
import utils
import tempfile
import accelerate
from diffuser import diff_utils as utils
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from absl import logging
from diffuser.diff_dataset import get_dataset
from diffuser import auto
from diffuser.fidscore import calculate_fid_given_paths
from diffuser.dpm import DPM_Solver, NoiseScheduleVP





def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True,
                                     drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True)

    nnet = utils.get_nnet(**config.nnet)
    nnet, test_dataset_loader = accelerator.prepare(nnet, test_dataset_loader)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu')) # 获取基础的模型对象
    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        # classifier-free
        _cond = nnet(x, timesteps, context=context)
        if config.sample.scale == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    autoencoder = auto.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                yield _context

    context_generator = get_context_generator()

    _betas = utils.stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode=t2i, mixed_precision={config.mixed_precision}')

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)
        return decode_large_batch(_z)

    def sample_fn(_n_samples):
        # 利用context生成目标样本
        _context = next(context_generator)
        assert _context.size(0) == _n_samples
        return dpm_solver_sample(_n_samples, config.sample.sample_steps, context=_context)

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'nnet_path={config.nnet_path}, fid={fid}') # 日志不会少，但是tem_path会少
