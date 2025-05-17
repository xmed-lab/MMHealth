# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : models.py
# Time       ：4/11/2024 3:24 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：模型定义 MedDiff
"""
import torch
import numpy as np
import torch.nn as nn

import itertools
import time
import pandas as pd
import torch
import dgl
import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional, Union
# from pyhealth.models.utils import get_last_visit
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.functional import multilabel_margin_loss
from pyhealth.metrics import ddi_rate_score
from pyhealth.models.utils import batch_to_multihot
from pyhealth.models import BaseModel
from pyhealth.medcode import ATC
from pyhealth.datasets import SampleEHRDataset
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from utils import get_last_visit
from utils import pad_list


class PredJoint(nn.Module):
    def __init__(self, feature_num, embedding_dim, nhead=2, dropout=0.1, voc_size=1000, task='REC'):
        super(PredJoint, self).__init__()
        rnns = torch.nn.TransformerEncoderLayer(
            d_model=feature_num * embedding_dim, nhead=nhead, batch_first=True, dropout=dropout)  # all others;
        self.rnns = torch.nn.TransformerEncoder(rnns, num_layers=1) # 有的任务，例如eicu为2
        self.fina_proj = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear((feature_num+1) * embedding_dim,
                                                 voc_size))
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)

        self.clz_token = nn.Parameter(torch.randn(1, feature_num*embedding_dim))

    def add_clz_mask(self, mask):
        # 在掩码前添加 CLZ token
        batch_size, seq_length = mask.shape
        clz_mask = torch.ones(batch_size, 1).to(mask.device)  # CLZ token 的掩码为 1
        return torch.cat((clz_mask, mask), dim=1).bool()

    def add_clz_token(self, batch_data, clz_token):
        # 在每个句子前添加 CLZ token
        batch_size, seq_length, embed_size = batch_data.shape  # B,V,D
        # 创建 CLZ token的形状为 (B, 1, D)
        clz_token = clz_token.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 CLZ token 和原始数据
        return torch.cat((clz_token, batch_data), dim=1)

    def forward(self, patient_emb, mask, id=None, new_feature=None):
        mask = self.add_clz_mask(mask)
        patient_emb = self.add_clz_token(patient_emb, self.clz_token)

        x = self.rnns(patient_emb, src_key_padding_mask=~mask)

        x = get_last_visit(x, mask)
        x = torch.cat([id, x], dim=-1)

        if new_feature is not None:
            x = x + new_feature.view(new_feature.size(0),-1)
        logits = self.fina_proj(x)
        y_prob = self.final_act(logits)
        return logits, y_prob





class RecLayer(nn.Module):
    # multi emb inverse
    def __init__(self, embedding_dim, hidden_dim, voc_size, feature_key, dropout=0.1, nhead=2, config=None):
        super(RecLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ddi_weight = 0.
        self.multiloss_weight = config['MULTI']

        self.feature_key = feature_key
        self.feature_num = len(feature_key)

        self.id_proj = nn.Sequential(
            nn.Linear(self.feature_num * embedding_dim, embedding_dim, bias=False),
            )
        gru = nn.TransformerEncoderLayer(d_model=3*embedding_dim, nhead=4, dropout=dropout, batch_first=True)
        self.gru = nn.TransformerEncoder(gru, num_layers=1)

        self.config=config
        # self.pred_layer_2 = PredSingle(embedding_dim, nhead=nhead, dropout=dropout,voc_size=voc_size)
        # self.pred_layer_3 = PredSingle(embedding_dim, nhead=nhead, dropout=dropout,voc_size=voc_size)
        self.pred_joint = PredJoint(self.feature_num, embedding_dim, nhead=4, dropout=0.3,voc_size=voc_size, task=self.config['TASK'])
        self.beta = np.ones((self.feature_num,))
        self.eta = 0.001 # decay


    def forward(
            self,
            patient_id: torch.Tensor,
            patient_emb: torch.Tensor,
            labels: torch.Tensor,
            ddi_adj: torch.Tensor,
            mask: torch.Tensor,
            labels_indexes: Optional[torch.Tensor] = None,
            soft_feature: torch.Tensor = None,
        ):
        patient_id = self.gru(patient_id, src_key_padding_mask=~mask)
        patient_id = get_last_visit(patient_id, mask)
        patient_id = self.id_proj(patient_id)


        joint_emb = torch.cat([patient_emb[feature] for feature in self.feature_key], dim=-1)
        # joint_emb = torch.cat([patient_id, joint_emb], dim=-1)
        logit_joint, y_prob_joint = self.pred_joint(joint_emb, mask, patient_id) # soft_feature
        loss_joint = self.calculate_loss(logit_joint,  y_prob_joint, labels, labels_indexes)
        return loss_joint, y_prob_joint


    def calculate_loss(
            self,
            logits: torch.Tensor,
            y_prob: torch.Tensor,
            labels: torch.Tensor,
            label_index: Optional[torch.Tensor] = None,
    ):
        # mul_pred_prob = y_prob.T @ y_prob  # B,D-> B,D (voc_size, voc_size)，感觉这里会有问题。过大的batchsize会有问题。
        # ddi_loss = 0 # (mul_pred_prob * ddi_adj).sum() / (ddi_adj.shape[0] ** 2) # 这里感觉也有问题，是乘以固定值 0.0005

        # y_pred = y_prob.detach().cpu().numpy()
        # y_pred[y_pred >= 0.5] = 1
        # y_pred[y_pred < 0.5] = 0 # 优化同步
        # y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        if self.config['TASK'] in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            loss_cls = binary_cross_entropy_with_logits(logits, labels)
        elif self.config['TASK'] in ['LOS']:
            loss_cls = F.cross_entropy(logits, labels)

        if self.multiloss_weight > 0 and label_index is not None:
            loss_multi = multilabel_margin_loss(y_prob, label_index)
            loss_cls = self.multiloss_weight * loss_multi + (1 - self.multiloss_weight) * loss_cls

        return loss_cls


class MedDiff(BaseModel):
    # pyhealth通用接口
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys=["conditions", "procedures", "drugs", "incomplete"],
            label_key="labels",
            mode="multilabel",

            # hyper related
            dropout: float = 0.3,
            num_rnn_layers: int = 2,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            **kwargs,
    ):
        super(MedDiff, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        # define
        self.num_rnn_layers = num_rnn_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.config = kwargs['config']

        self.dropout_id = torch.nn.Dropout(self.dropout)


        self.feat_tokenizers = self.get_feature_tokenizers()  # tokenizer
        self.label_tokenizer = self.get_label_tokenizer()  # 注意这里的drug可没有spec_token; 这里label索引需要加2对于正则化
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        # save ddi adj
        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
        ddi_adj = self.generate_ddi_adj()  # 用于存储
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())  # 计算ddi直接从这里读取

        # module
        self.feature_keys_subs = ['conditions', 'procedures', 'drugs'] if self.config['DATASET'] != 'MIV-Note' else ['conditions', 'procedures', 'drugs', 'note']
        self.rec_layer = RecLayer(self.embedding_dim, self.hidden_dim, self.label_size,
                                  feature_key=self.feature_keys_subs, dropout=dropout, config=self.config)
        self.rnns = nn.ModuleDict({x: nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.num_rnn_layers, batch_first=True) for x in self.feature_keys_subs})
        # self.diff_proj = nn.Sequential(nn.Dropout(0.8),nn.Linear(self.embedding_dim, self.embedding_dim))
        self.diff_proj = nn.Sequential(nn.Dropout(0.1),nn.Linear(2*self.embedding_dim, self.embedding_dim))
        # 特殊/tmp定义， 不能放在init_weights之前，会使得embedding机制失效
        # init params
        # self.apply(self.init_weights)

        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)  # ehr emb

        # new for note dataset
        self.note_linear = nn.Linear(768, self.embedding_dim)

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0.1)  # He initialization for Conv2d
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for Linear layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero

    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True) # dataframe，这里使用了gamenet的ddi,不要存储
        # ddi = pd.read_csv('/home/czhaobo/KnowHealth/data/REC/MIII/processed/ddi_pairs.csv', header=0, index_col=0).values.tolist()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi # each row
        ]

        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def encode_patient(self, feature_key: str, raw_values: List[List[List[str]]], new_feature=0) -> torch.Tensor:
        """Encode patient data."""
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(raw_values,
                                                                  max_length=[self.config['MAXSEQ'],
                                                                              self.config['MAXCODESEQ']])  # 这里会padding, B,V,M
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        masks = codes != 0  # B,V,M
        embeddings = self.embeddings[feature_key](codes)  # B,V,M,D
        embeddings = self.dropout_id(embeddings)
        visit_emb = self.get_visit_emb(embeddings, feature_key, masks)  # B,V,D


        # new hard sample
        if new_feature:
            if self.config['MODEL'] == 'ours':
                new_codes = self.feat_tokenizers[feature_key].batch_encode_3d(new_feature,
                                                                          max_length=[self.config['MAXSEQ'],
                                                                                      self.config[
                                                                                          'MAXCODESEQ']])  # 这里会padding, B,V,M
                new_codes = torch.tensor(new_codes, dtype=torch.long, device=self.device)
                #masks = new_codes != 0  # B,V,M
                new_embeddings = self.embeddings[feature_key](new_codes)  # B,V,M,D
                new_embeddings = self.dropout_id(new_embeddings)
                new_visit_emb = self.get_visit_emb(new_embeddings)  # B,V,D
            elif self.config['MODEL']=='MedDiff':
                new_visit_emb = torch.tensor(new_feature, dtype=torch.float, device=self.device).unsqueeze(dim=1) # B,D
                new_visit_emb = F.normalize(new_visit_emb, p=2, dim=-1)  # B,D
                # print("BBBBB",new_visit_emb.shape)
            # visit_emb = visit_emb + new_visit_emb
            visit_emb = torch.cat([visit_emb, new_visit_emb.repeat(1,visit_emb.shape[1],1)], dim=-1) # 不行就只能viisit_emb
            visit_emb = self.diff_proj(visit_emb)


        return codes, embeddings, masks, visit_emb  # B,V, D



    def get_visit_emb(self, emb, feature_key=None, masks=None):
        """Get visit embedding."""
        # b, v, m, d = emb.shape
        # emb = emb.view(b*v, m, d)
        # emb,_ = self.rnns[feature_key](emb)
        # masks = masks.view(b*v, m)
        # emb = get_last_visit(emb, masks)
        # emb = emb.view(b, v, d)

        emb = torch.sum(emb, dim=2)
        return emb

    def decode_label(self, array_prob, tokenizer):
        """给定y概率，label tokenizer，返回所解码出来的code Token"""
        array_prob[array_prob >= 0.4] = 1
        array_prob[array_prob < 0.4] = 0  # 优化同步
        indices = [np.where(row == 1)[0].tolist() for row in array_prob]
        tokens = tokenizer.batch_decode_2d(indices)
        return tokens

    def forward(
            self,
            patient_id: List[List[str]],
            conditions: List[List[List[str]]],  # 需要和dataset保持一致[名字，因为**data]
            procedures: List[List[List[str]]],
            drugs_hist: List[List[List[str]]],
            labels: List[List[str]],  # label
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels]
                    representing the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels]
                    representing the ground truth of each drug.
        """
        drugs = labels  # for case
        # # patient id
        # prepare labels
        if self.mode == "multilabel":
            labels_index = self.label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )  # [[23,32],[1,2,3]]，注意比feature_tokenizer少两位

            labels = batch_to_multihot(labels_index, self.label_size)  # tensor, B, Label_size;  # convert to multihot
            index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
            for idx, cont in enumerate(labels_index):
                # remove redundant labels
                cont = list(set(cont))
                index_labels[idx, : len(cont)] = cont  # remove padding and unk
            index_labels = torch.from_numpy(index_labels)  # 类似的！【23，38，39】
            labels = labels.to(self.device)  # for bce loss
            index_labels = index_labels.to(self.device)  # for multi label loss
        elif self.mode in ["multiclass", 'binary']:
            index_labels = None
            labels = self.prepare_labels(labels, self.label_tokenizer)



        # patient id
        # kwargs['miss_feature'] = torch.tensor(kwargs['miss_feature']).to(self.device)
        cond_code, _, condi_mask, condition_vis_emb = self.encode_patient("conditions",
                                                                          conditions, kwargs['new_conditions'])  # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
        proc_code, _, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures, kwargs['new_procedures'])
        drug_code, _, drug_mask, drug_history_vis_emb = self.encode_patient("drugs",
                                                                            drugs_hist, kwargs['new_drugs'])  # drug rec的时候不能放drug 1，1，1，1

        # emb,_ = self.rnns[feature_key](emb)
        # masks = masks.view(b*v, m)
        # emb = get_last_visit(emb, masks)
        # condition_vis_emb = self.rnns['conditions'](condition_vis_emb)[0]
        # procedure_vis_emb = self.rnns['procedures'](procedure_vis_emb)[0]
        # drug_history_vis_emb = self.rnns['drugs'](drug_history_vis_emb)[0]


        # new for note
        if self.config['DATASET'] == 'MIV-Note':
            note_emb = self.note_linear(pad_list(kwargs['note'], device=self.device)) # B,T,D
            note_mask = note_emb.sum(dim=-1) != 0 # B,T

            seq_emb = {'conditions': condition_vis_emb, 'procedures': procedure_vis_emb, 'drugs': drug_history_vis_emb, 'note': note_emb}
            mask = (torch.sum(condi_mask, dim=-1) + torch.sum(proc_mask, dim=-1) + torch.sum(drug_mask, dim=-1) + note_mask) != 0 # torch.sum(condi_mask, dim=-1) != 0  # visit-level mask; 这个更安全，emb相加可能为0
        else:
            seq_emb = {'conditions': condition_vis_emb, 'procedures': procedure_vis_emb, 'drugs': drug_history_vis_emb}
            mask = (torch.sum(condi_mask, dim=-1) + torch.sum(proc_mask, dim=-1) + torch.sum(drug_mask, dim=-1)) != 0

        patient_emb = seq_emb

        patient_id = torch.cat(
            [seq_emb[feature] for feature in self.feature_keys_subs], dim=-1)

        # calculate loss
        loss, y_prob = self.rec_layer(  # patient
            patient_id,
            patient_emb=patient_emb,
            labels=labels,
            ddi_adj=self.ddi_adj,
            mask=mask,
            labels_indexes=index_labels,
            soft_feature=None,
        )

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "drugs": drugs,  # case时候打开
        }
