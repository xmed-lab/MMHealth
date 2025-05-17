# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : diff_utils.py
# Time       ：5/11/2024 3:03 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：diffusion的一些工具函数
"""
import random

# code from timm 0.3.2

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from absl import logging
from .unitrans import DIT
import torch.nn.functional as F

from pyhealth.models.utils import batch_to_multihot
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from collections import defaultdict


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(num_train_visit, tokenizer, name, **kwargs):
    if name == 'dit':
        return DIT(num_train_visit, tokenizer, **kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    # 加权平均模型参数
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data) # ema更新


class TrainState(object):
    # 用于存储训练状态
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step # start step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        # 读取最新的ckpt
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters()) # * 4 // (1024**3)


def initialize_train_state(config, device, tokenizers, num_train_visit):
    params = []
    print("initial train visit", num_train_visit)
    nnet = get_nnet(num_train_visit, tokenizers, **config['nnet']) # 对应的model, 可以是unet或者Transformer
    params += nnet.parameters()

    nnet_ema = get_nnet(num_train_visit, tokenizers, **config['nnet'])
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config['optimizer'])
    lr_scheduler = get_lr_scheduler(optimizer, **config['lr_scheduler'])

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0) # 初始化ema
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    # 感觉可能也用不到
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]

def save_samples(samples, path):
    torch.save(samples, path)
    print("Save success!")

def calculate_quality(path):
    samples = torch.load(os.path.join(path, 'eval_samples.pth')) # 存储的是embeddings
    labels = torch.load(os.path.join(path , 'eval_labels.pth'))
    hard_samples = torch.load(os.path.join(path, 'hard_samples.pth'))
    hard_labels = torch.load(os.path.join(path, 'hard_labels.pth'))

    labels = labels.squeeze()
    # Ensure that the tensors are of the same shape
    if samples.shape != labels.shape:
        print(samples.shape, labels.shape)
        raise ValueError("Samples and labels must have the same shape.")

    # Calculate MSE loss
    mse_loss = F.mse_loss(samples, labels)

    # calculate the average Jaccard similarity
    # print("BBBBBBBBB", hard_samples['conditions'][0], hard_labels['conditions'][0])
    conditions_jaccard = calculate_average_jaccard(hard_samples['conditions'], hard_labels['conditions'])
    procedures_jaccard = calculate_average_jaccard(hard_samples['procedures'], hard_labels['procedures'])
    drugs_jaccard = calculate_average_jaccard(hard_samples['drugs'], hard_labels['drugs'])



    return mse_loss.item(), (conditions_jaccard,procedures_jaccard, drugs_jaccard)  # Return the loss as a Python float


def sample2dir_test(path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    # 存储采样结果
    os.makedirs(path, exist_ok=True)
    # idx = 0
    batch_size = mini_batch_size  # * accelerator.num_processes
    samples_list = defaultdict(list) # {'conditions':batch list, 'procedures':batch, 'drugs':batch}
    hard_sample_list = defaultdict(list) # {'conditions':batch list, 'procedures':batch, 'drugs':batch}
    
    for _batch_size in tqdm(amortize(n_samples, batch_size), desc='sample2dir'):  # 多取一个回合
        labels, _,  (samples, hard_samples) = sample_fn(_batch_size)  # 去噪后的结果, B,3,D， 感觉或许可以传rounding

        samples = unpreprocess_fn(samples) if unpreprocess_fn is not None else samples  # 这里一旦rounding，可能就需要改下denoise
        # samples_list.append(samples)
        for index, feature_key in enumerate(hard_samples.keys()):
            hard_sample_list[feature_key].extend(hard_samples[feature_key])
            samples_list[feature_key].extend(samples[:, index, :].detach().cpu().numpy().tolist())
    return samples_list, hard_sample_list # soft_feature, hard sample:


def sample2dir(path, n_samples, mini_batch_size, sample_fn, tokenizers, unpreprocess_fn=None):
    # 存储采样结果
    os.makedirs(path, exist_ok=True)
    # idx = 0
    batch_size = mini_batch_size #* accelerator.num_processes
    samples_list = []
    labels_list = []
    hard_samples_list = defaultdict(list) # {'conditions':batch list, 'procedures':batch, 'drugs':batch}
    hard_labels_list = defaultdict(list)
    for _batch_size in tqdm(amortize(n_samples, batch_size), desc='sample2dir'): # 多取一个回合
        labels, hard_labels, (samples, hard_samples) = sample_fn(_batch_size) # 去噪后的结果, B,3,D， 感觉或许可以传rounding
        samples = unpreprocess_fn(samples) if unpreprocess_fn is not None else samples # 这里一旦rounding，可能就需要改下denoise
        # samples = samples.contiguous()[:_batch_size]
        # if accelerator.is_main_process:
        # print("AAAAAAA", samples.shape)
        samples_list.append(samples) # B,3,D
        labels_list.append(labels)
        for feature_key in hard_samples.keys(): # {A:[[],[]]} len(A)=B
            hard_samples_list[feature_key].extend(hard_samples[feature_key])
            hard_labels_list[feature_key].extend(tokenizers[feature_key].batch_decode_2d(hard_labels[feature_key]))

    save_samples(torch.cat(samples_list, dim=0), os.path.join(path, "eval_samples.pth")) # soft
    save_samples(torch.cat(labels_list, dim=0), os.path.join(path, "eval_labels.pth"))
    save_samples(hard_samples_list, os.path.join(path, "hard_samples.pth")) # hard
    save_samples(hard_labels_list, os.path.join(path, "hard_labels.pth"))





def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


# diffusion的一些操作
def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    # beta = 0.00085 for 1000 steps, then linearly anneal to 0.012
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def q_mean_variance(self, x0):
        # 计算 cum_alphas 的平方根
        # alphas_sqrt = self.cum_alphas[:self.N] ** 0.5
        # betas_sqrt = self.cum_betas[:self.N] ** 0.5
        # # 计算均值
        # mean = torch.sum(stp(alphas_sqrt, x0), dim=0)
        # # 计算方差
        # variance = torch.sum(self.cum_alphas[1:self.N], dim=0)
        # # 构造噪声
        # noise = torch.randn_like(x0) * betas_sqrt.unsqueeze(1)
        n = np.ones(len(x0)) * (self.N-1)#torch.LongTensor(self.N-1).to(x0.device)
        mean = stp(self.cum_alphas[n.astype(int)] ** 0.5, x0)

        return mean

    def sample(self, x0, mask=None):
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        if mask is not None:
            # print(mask.shape, x0.shape, xn.shape)
            mask = mask.unsqueeze(-1).repeat(1, 1, x0.shape[-1])
            xn = torch.where(~mask, x0, xn) # no mask的地方用x0， 这里需要传入mask!!!

        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule, **kwargs):
    n, eps, xn = schedule.sample(x0, mask=kwargs['mask'])  # n in {1, ..., 1000}, step， error, xn; 【True, False】
    model_output = nnet(xn, n, kwargs['context']) # B,3,D
    # error estimate loss
    l1_loss = mos(eps - model_output, reduction=False) # B， 估计error

    # rounding loss
    get_logits = nnet.get_logits
    terms = {}
    terms["mse"] = mos(x0-model_output, reduction=False)
    loss_mse = torch.where(kwargs['mask'], terms["mse"], l1_loss).mean(dim=-1)

    terms['token'] = token_loss(x0, get_logits, kwargs, nnet.feature_keys, nnet.tokenizer) # x0的。
    terms['nll'] = token_loss(x0, get_logits, kwargs, nnet.feature_keys, nnet.tokenizer, is_mask=False)

    # mean均衡损失
    out_mean = schedule.q_mean_variance(x0)
    tT_loss = mean_flat(out_mean ** 2)

    loss_mse = loss_mse

    loss_round = terms['token'] + terms['nll']
    return loss_mse, loss_round, tT_loss


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def token_loss(x_t, get_logits, raw_data, feature_keys, tokenizers, is_mask=True):
    loss_fct = binary_cross_entropy_with_logits # torch.nn.CrossEntropyLoss(reduction='none')

    mask = raw_data['mask'] # B, 3
    reshaped_x_t = x_t # B,3,D

    # binary labels
    decoder_nll = []
    for index, feature_key in enumerate(feature_keys):
        labels = batch_to_multihot(raw_data[feature_key + '_comps'], len(tokenizers[feature_key].vocabulary))  # tensor, B, Label_size;  # convert to multihot
        labels = labels[:, 2:] # 保持大小, ywei
        labels = labels.to(reshaped_x_t.device)  # for bce loss
        
        logits = get_logits(reshaped_x_t[:,index,:], feature_key)  # bsz, voc
        loss_index = loss_fct(logits, labels, reduction='none').mean(dim=1).unsqueeze(dim=1) # wrong
        decoder_nll.append(loss_index)

    decoder_nll = torch.cat(decoder_nll, dim=-1) # B,3
    # print("AAAAAA", decoder_nll)
    # print("CCCCC", raw_data['mask'])
    if mask != None and is_mask: # mask位置的解码损失
        decoder_nll *= mask  # 去掉~mask部分的token loss
        decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
        # print("BBBBBB", decoder_nll)
        decoder_nll = decoder_nll
    else:
        decoder_nll = decoder_nll.mean(dim=-1)

    return decoder_nll


def stp(s, ts: torch.Tensor):  # scalar tensor product,zhu元素dot。
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1, reduction=True):  # mean of square
    if reduction:
        return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)
    else:
        return a.pow(2).mean(dim=-1) # B,3


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas




def get_last_visit_sample(samples):
    last_visits = {}
    for record in samples:
        patient_id = record['patient_id']
        visit_id = int(record['visit_id'])  # 将visit_id转换为整数
        if patient_id not in last_visits or visit_id > int(last_visits[patient_id]['visit_id']):
            last_visits[patient_id] = record
    print("Patient Number: ", len(last_visits))
    return last_visits

######## others
def calculate_average_jaccard(list_a, list_b):
    def jaccard_index(a, b):
        set_a = set(a)
        set_b = set(b)

        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)

        return len(intersection) / len(union) if len(union) > 0 else 0

    # 计算每对子列表的 Jaccard 相似度
    jaccard_results = [jaccard_index(a, b) for a, b in zip(list_a, list_b)]

    # 计算平均 Jaccard 相似度
    average_jaccard = sum(jaccard_results) / len(jaccard_results) if jaccard_results else 0

    return average_jaccard


def generate_mask(batch_size, m=3, mask_num=1):
    # 随机生成每一行的掩码列索引
    indices = torch.randint(0, m, (batch_size, mask_num))
    # 创建掩码矩阵
    mask = torch.zeros((batch_size, m), dtype=torch.int)
    # 使用高级索引设置掩码
    mask[torch.arange(batch_size).unsqueeze(1), indices] = 1
    mask = mask.bool()

    return mask # mask | mask2

