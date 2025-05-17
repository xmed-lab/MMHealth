# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : transformer.py
# Time       ：17/12/2024 3:01 pm
# Author     ：Chuang Zhao
# version    ：python
# Description：
"""




import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyhealth.models import BaseModel
from pyhealth.medcode import ATC
from pyhealth.datasets import SampleEHRDataset
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from utils import get_last_visit,pad_list
from typing import Any, Dict, List, Tuple, Optional, Union
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.functional import multilabel_margin_loss
from pyhealth.models.utils import batch_to_multihot
import torch.nn.functional as F

class RecLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, voc_size, feature_num=3, dropout=0.1, nhead=2, config=None):
        super(RecLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ddi_weight = 0.
        self.multiloss_weight = config['MULTI']
        self.feature_num = feature_num

        self.id_proj = nn.Sequential(nn.Linear(self.feature_num* embedding_dim, self.feature_num* embedding_dim, bias=False),
                                       )
        self.rnns = torch.nn.TransformerEncoderLayer(
                    d_model=self.feature_num*embedding_dim, nhead=nhead, batch_first=True)
        self.fina_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_num * embedding_dim, voc_size))

        self.config=config
        task = config['TASK']
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)

    def forward(
            self,
            patient_id: torch.Tensor,
            patient_emb: torch.Tensor,
            labels: torch.Tensor,
            ddi_adj: torch.Tensor,
            mask: torch.Tensor,
            labels_indexes: Optional[torch.Tensor] = None,
    ):
        # patient_emb = torch.cat([patient_id, patient_emb], dim=1) # B, 4D
        # print(patient_emb.shape)
        patient_id = self.id_proj(patient_id) # B, D, diag这个玩意很有作用

        patient_emb = self.rnns(patient_emb, src_key_padding_mask=~mask)  # B,V,D
        patient_emb = get_last_visit(patient_emb, mask)

        # patient_emb = torch.cat([patient_id, patient_emb], dim=1)  # B, 2D

        logits = self.fina_proj(patient_emb)  # B, Label_size
        y_prob = self.final_act(logits)
        loss = self.calculate_loss(logits, y_prob, ddi_adj, labels, labels_indexes)
        return loss, y_prob

    def calculate_loss(
            self,
            logits: torch.Tensor,
            y_prob: torch.Tensor,
            ddi_adj: torch.Tensor,
            labels: torch.Tensor,
            label_index: Optional[torch.Tensor] = None,
    ):

        if self.config['TASK'] in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            loss_cls = binary_cross_entropy_with_logits(logits, labels)
        elif self.config['TASK'] in ['LOS']:
            loss_cls = F.cross_entropy(logits, labels)

        if self.multiloss_weight > 0 and label_index is not None:
            loss_multi = multilabel_margin_loss(y_prob, label_index)
            loss_cls = self.multiloss_weight * loss_multi + (1 - self.multiloss_weight) * loss_cls

        return loss_cls








class Transformer(BaseModel):
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
        super(Transformer, self).__init__(
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
        self.label_tokenizer = self.get_label_tokenizer()
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        # save ddi adj
        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
        ddi_adj = self.generate_ddi_adj()  # 用于存储
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())  # 计算ddi直接从这里读取

        # module
        self.feature_keys_subs = ['conditions', 'procedures', 'drugs']
        self.rec_layer = RecLayer(self.embedding_dim, self.hidden_dim, self.label_size,
                                  feature_num=len(self.feature_keys_subs), dropout=dropout, config=self.config)

        # init params
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)  # ehr emb

        # note
        self.note_linear = nn.Linear(768, self.embedding_dim)




    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
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
                                                                              self.config['MAXCODESEQ']])
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        masks = codes != 0  # B,V,M
        embeddings = self.embeddings[feature_key](codes)
        embeddings = self.dropout_id(embeddings)
        visit_emb = self.get_visit_emb(embeddings)


        # new hard sample
        if new_feature:
            new_codes = self.feat_tokenizers[feature_key].batch_encode_3d(new_feature,
                                                                      max_length=[self.config['MAXSEQ'],
                                                                                  self.config[
                                                                                      'MAXCODESEQ']])  # 这里会padding, B,V,M
            new_codes = torch.tensor(new_codes, dtype=torch.long, device=self.device)
            #masks = new_codes != 0  # B,V,M
            new_embeddings = self.embeddings[feature_key](new_codes)  # B,V,M,D
            new_embeddings = self.dropout_id(new_embeddings)
            new_visit_emb = self.get_visit_emb(new_embeddings)  # B,V,D
            visit_emb = visit_emb + new_visit_emb

        return codes, embeddings, masks, visit_emb  # B,V, D

    def get_visit_emb(self, emb, feature_key=None, masks=None):
        """Get visit embedding."""
        emb = torch.sum(emb, dim=2)
        # emb = self.dropout_id(torch.sum(emb, dim=2))
        return emb

    def decode_label(self, array_prob, tokenizer):
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
        if self.mode == "multilabel":
            labels_index = self.label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )

            labels = batch_to_multihot(labels_index, self.label_size)  # tensor, B, Label_size;  # convert to multihot
            index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
            for idx, cont in enumerate(labels_index):
                # remove redundant labels
                cont = list(set(cont))
                index_labels[idx, : len(cont)] = cont
            index_labels = torch.from_numpy(index_labels)
            labels = labels.to(self.device)  # for bce loss
            index_labels = index_labels.to(self.device)  # for multi label loss
        elif self.mode in ["multiclass", 'binary']:
            index_labels = None
            labels = self.prepare_labels(labels, self.label_tokenizer)


        cond_code, _, condi_mask, condition_vis_emb = self.encode_patient("conditions",
                                                                          conditions)  # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
        proc_code, _, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures)
        drug_code, _, drug_mask, drug_history_vis_emb = self.encode_patient("drugs",
                                                                            drugs_hist)  # drug rec的时候不能放drug 1，1，1，1

        # mask = torch.sum(condi_mask, dim=-1) != 0  # visit-level mask; 这个更安全，emb相加可能为0
        # print(mask.sum(dim=1))
        # patient_emb = torch.cat([seq_emb[feature] for feature in self.feature_keys_subs], dim=2)  # B,V,3D
        # patient_emb = [seq_emb[feature] for feature in self.feature_keys_subs]  # B,V,3D

        # new for note

        if self.config['DATASET']=='MIV-Note':
            note_emb = self.note_linear(pad_list(kwargs['note'], device=self.device))  # B,T,D
            note_mask = note_emb.sum(dim=-1) != 0  # B,T
            condition_vis_emb = condition_vis_emb + note_emb
            condi_mask = (torch.sum(condi_mask, dim=-1) +note_mask)!=0
        else:
            condi_mask = torch.sum(condi_mask, dim=-1) != 0  # visit-level mask B,T; [True, True, False]



        mask = (condi_mask + torch.sum(proc_mask, dim=-1) + torch.sum(drug_mask, dim=-1)) != 0 # visit-level mask B,T; [True, True, False]


        # patient_emb = torch.cat([condition_vis_emb, procedure_vis_emb, drug_history_vis_emb], dim=2)  # B,V,3D
        seq_emb = {'conditions': condition_vis_emb, 'procedures': procedure_vis_emb, 'drugs': drug_history_vis_emb} # B,V,D
        seq_mask = {'conditions': condi_mask, 'procedures':  torch.sum(proc_mask, dim=-1)!=0, 'drugs':  torch.sum(drug_mask, dim=-1)!=0} 
        
        patient_id = torch.cat([self.dropout_id(get_last_visit(seq_emb[feature], mask)) for feature in self.feature_keys_subs],dim=1)

        patient_emb = torch.cat([seq_emb[feature] for feature in self.feature_keys_subs], dim=2)  # B,V,3D

        # calculate loss
        loss, y_prob = self.rec_layer(  # patient
            patient_id,
            patient_emb=patient_emb,
            labels=labels,
            ddi_adj=self.ddi_adj,
            mask=mask,
            labels_indexes=index_labels,
        )


        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "drugs": drugs,  # case时候打开
        }


