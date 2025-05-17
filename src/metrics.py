# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : metrics.py
# Time       ：12/11/2024 7:06 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：ack pyhealth
"""


from typing import Dict, List, Optional, Any
import os

import numpy as np
import sklearn.metrics as sklearn_metrics
from pyhealth.medcode import ATC
import pyhealth.metrics.calibration as calib
from pyhealth.metrics import ddi_rate_score
from pyhealth import BASE_CACHE_PATH as CACHE_PATH

from typing import Dict, List, Optional

import numpy as np
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
import pyhealth.metrics.prediction_set as pset
from config import config
from collections import Counter


def group_rec(y, p, pred, patient_ids, group_p): # 这个或许可以扩展到其他的multi-class啥的
    # jaccard, f1-score, precision, recall, roc_auc, pr_auc
    # print(group_p)
    # print("==========")
    # print(patient_ids) # ['85424', '2613', '4931', '90917']
    # print(group_p)
    # print("AAAAAA", patient_ids)
    patient_ids = np.array([group_p[id] for id in patient_ids]) # rare type
    zero_indices = np.where(patient_ids == 'G1')[0]
    one_indices = np.where(patient_ids == 'G2')[0]
    two_indices = np.where(patient_ids == 'G3')[0]
    three_indices = np.where(patient_ids == 'G4')[0] # 缺失的最少
    group_indices = {'G1':zero_indices, 'G2':one_indices, 'G3':two_indices, 'G4':three_indices}
    out = {}
    for key, lis in group_indices.items():
        y_g, p_g, pred_g = y[lis], p[lis], pred[lis]
        # print(y_g.shape, p_g.shape)
        roc_auc_samples = sklearn_metrics.roc_auc_score(
            y_g, p_g, average="samples"
        )
        pr_auc_samples = sklearn_metrics.average_precision_score(
                y_g, p_g, average="samples"
            )
        f1_samples = sklearn_metrics.f1_score(y_g, pred_g, average="samples",  zero_division=1)
        jaccard_samples = sklearn_metrics.jaccard_score(
            y_g, pred_g, average="samples", zero_division=1
        )
        out[key] = [jaccard_samples, f1_samples, pr_auc_samples, roc_auc_samples]
    return out


def group_cls(y, p, pred, patient_ids, group_p): # 这个或许可以扩展到其他的multi-class啥的
    patient_ids = np.array([group_p[id] for id in patient_ids]) # rare type
    zero_indices = np.where(patient_ids == 'G1')[0]
    one_indices = np.where(patient_ids == 'G2')[0]
    two_indices = np.where(patient_ids == 'G3')[0]
    three_indices = np.where(patient_ids == 'G4')[0]
    group_indices = {'G1':zero_indices, 'G2':one_indices, 'G3':two_indices, 'G4':three_indices}
    out = {}
    for key, lis in group_indices.items():
        y_g, p_g, pred_g = y[lis], p[lis], pred[lis]
        cohen_kappa = sklearn_metrics.cohen_kappa_score(y_g, pred_g,)
        accuracy = sklearn_metrics.accuracy_score(y_g, pred_g,)
        f1_weighted = sklearn_metrics.f1_score(y_g, pred_g, average="weighted")
        # roc_auc_weighted_ovr = sklearn_metrics.roc_auc_score(
        #     y_g, p_g, average="weighted", multi_class="ovr"
        # )
        out[key] = [cohen_kappa, accuracy,f1_weighted]
    return out





def topk_precision(y, p, k=20):
    """Computes precision at k for multilabel classification."""
    ret_lst = []
    for i in range(y.shape[0]):
        predictions = np.argsort(p[i, :])[-k:]
        true_labels = np.nonzero(y[i, :])[0]
        n_correct = np.in1d(true_labels, predictions, assume_unique=True).sum()  # 直接计算precision
        # pdb.set_trace()

        ret_lst.append(n_correct / min(len(true_labels), k))

    return np.mean(ret_lst)


def topk_acc(y, p, k, grouped_y=None):
    """Computes top-k accuracy for multilabel classification."""
    if grouped_y is None:
        total_counter = Counter()
        correct_counter = Counter()
        for i in range(y.shape[0]):
            true_labels = np.nonzero(y[i, :])[0]
            predictions = np.argsort(p[i, :])[-k:]
            for l in true_labels:
                total_counter[l] += 1
                correct_counter[l] += np.in1d(l, predictions, assume_unique=True).sum()

        total_labels = sum(total_counter.values())
        correct_labels = sum(correct_counter.values())
        acc_at_k = correct_labels / total_labels
        acc_at_k_grouped = {'nogroup': acc_at_k}
    else:
        # 分组check
        total_counter = Counter()
        correct_counter = Counter()

        for i in range(y.shape[0]):
            true_labels = np.nonzero(y[i, :])[0]  # 真实的[1,0,1,0,1]->[32,33,46]
            predictions = np.argsort(p[i, :])[-k:]  # topk [10,9,8]
            for l in true_labels:
                total_counter[l] += 1 # 真正出现的次数
                correct_counter[l] += np.in1d(l, predictions, assume_unique=True).sum()  # 预测的次数，如果存在则加1

        y_grouped = grouped_y  # {'10':[32,33,34]}
        n_groups = len(y_grouped)
        total_labels = [0] * n_groups  # 每个组别。
        correct_labels = [0] * n_groups
        for i, group in enumerate(y_grouped): # 以组为单位计数
            for l in group:
                correct_labels[i] += correct_counter[l]
                total_labels[i] += total_counter[l]

        acc_at_k_grouped = [x / float(y) for x, y in zip(correct_labels, total_labels)]  # grouped
        acc_at_k = sum(correct_labels) / float(sum(total_labels))  # all

    return acc_at_k, acc_at_k_grouped


def regression_metrics_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
    aux_data: Optional[Dict[str, Any]] = None,
    patient_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Computes metrics for regression.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - mse: mean squared error
        - rmse: root mean squared error
        - mae: mean absolute error
        - r2: R squared

    If no metrics are specified, mse is computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_pred: Predicted target values of shape (n_samples,).
        metrics: List of metrics to compute. Default is ["mse"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import regression_metrics_fn
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.1, 3.1])
        >>> regression_metrics_fn(y_true, y_pred, metrics=["mse"])
        {'mse': 0.01}
    """
    if metrics is None:
        metrics = ["mse"]

    output = {}
    for metric in metrics:
        if metric == "mse":
            mse = sklearn_metrics.mean_squared_error(y_true, y_pred)
            output["mse"] = mse
        elif metric == "rmse":
            rmse = np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))
            output["rmse"] = rmse
        elif metric == "mae":
            mae = sklearn_metrics.mean_absolute_error(y_true, y_pred)
            output["mae"] = mae
        elif metric == "r2":
            r2 = sklearn_metrics.r2_score(y_true, y_pred)
            output["r2"] = r2
        else:
            raise ValueError(f"Unknown metric for regression: {metric}")
    return output


def multiclass_metrics_fn(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metrics: Optional[List[str]] = None,
        y_predset: Optional[np.ndarray] = None,
        aux_data: Optional[Dict[str, Any]] = None,
        patient_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_micro"]
    prediction_set_metrics = [
        "rejection_rate",
        "set_size",
        "miscoverage_mean_ps",
        "miscoverage_ps",
        "miscoverage_overall_ps",
        "error_mean_ps",
        "error_ps",
        "error_overall_ps",
    ]
    y_pred = np.argmax(y_prob, axis=-1)

    output = {}
    for metric in metrics:
        if metric == "roc_auc_macro_ovo":
            roc_auc_macro_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovo"
            )
            output["roc_auc_macro_ovo"] = roc_auc_macro_ovo
        elif metric == "roc_auc_macro_ovr":
            roc_auc_macro_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            )
            output["roc_auc_macro_ovr"] = roc_auc_macro_ovr
        elif metric == "roc_auc_weighted_ovo":
            roc_auc_weighted_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovo"
            )
            output["roc_auc_weighted_ovo"] = roc_auc_weighted_ovo
        elif metric == "roc_auc_weighted_ovr":
            roc_auc_weighted_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovr"
            )
            output["roc_auc_weighted_ovr"] = roc_auc_weighted_ovr
        ##### rebuattl
        elif metric == "rmse":
            rmse = np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))
            output["rmse"] = rmse
        elif metric == "mae":
            mae = sklearn_metrics.mean_absolute_error(y_true, y_pred)
            output["mae"] = mae
        ##### rebuattl


        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro")
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "group_cls":
            output['cls_grouped'] = group_cls(y_true, y_prob, y_pred, patient_ids, group_p=aux_data['p_grouped'])

        elif metric == "jaccard_micro":
            jacard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jacard_micro
        elif metric == "jaccard_macro":
            jacard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jacard_macro
        elif metric == "jaccard_weighted":
            jacard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jacard_weighted
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "brier_top1":
            output[metric] = calib.brier_top1(y_prob, y_true)
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = calib.ece_confidence_multiclass(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
            )
            
        elif metric in {"cwECEt", "cwECEt_adapt"}:
            thres = min(0.01, 1.0 / y_prob.shape[1])
            output[metric] = calib.ece_classwise(
                y_prob,
                y_true,
                bins=20,
                adaptive=metric.endswith("_adapt"),
                threshold=thres,
            )
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == "rejection_rate":
                output[metric] = pset.rejection_rate(y_predset)
            elif metric == "set_size":
                output[metric] = pset.size(y_predset)
            elif metric == "miscoverage_mean_ps":
                output[metric] = pset.miscoverage_ps(y_predset, y_true).mean()
            elif metric == "miscoverage_ps":
                output[metric] = pset.miscoverage_ps(y_predset, y_true)
            elif metric == "miscoverage_overall_ps":
                output[metric] = pset.miscoverage_overall_ps(y_predset, y_true)
            elif metric == "error_mean_ps":
                output[metric] = pset.error_ps(y_predset, y_true).mean()
            elif metric == "error_ps":
                output[metric] = pset.error_ps(y_predset, y_true)
            elif metric == "error_overall_ps":
                output[metric] = pset.error_overall_ps(y_predset, y_true)

        elif metric == "hits@n":
            argsort = np.argsort(-y_prob, axis=1)
            ranking = np.array([np.where(argsort[i] == y_true[i])[0][0] for i in range(len(y_true))]) + 1
            output["HITS@1"] = np.count_nonzero(ranking <= 1) / len(ranking)
            output["HITS@5"] = np.count_nonzero(ranking <= 5) / len(ranking)
            output["HITS@10"] = np.count_nonzero(ranking <= 10) / len(ranking)
        elif metric == "mean_rank":
            argsort = np.argsort(-y_prob, axis=1)
            ranking = np.array([np.where(argsort[i] == y_true[i])[0][0] for i in range(len(y_true))]) + 1
            mean_rank = np.mean(ranking)
            mean_reciprocal_rank = np.mean(1 / ranking)
            output["mean_rank"] = mean_rank
            output["mean_reciprocal_rank"] = mean_reciprocal_rank

        else:
            raise ValueError(f"Unknown metric for multiclass classification: {metric}")

    return output


def multilabel_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = config['THRES'],
    y_predset: Optional[np.ndarray] = None,
    aux_data: Optional[Dict[str, Any]] = None,
    patient_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes metrics for multilabel classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - roc_auc_micro: area under the receiver operating characteristic curve,
          micro averaged
        - roc_auc_macro: area under the receiver operating characteristic curve,
          macro averaged
        - roc_auc_weighted: area under the receiver operating characteristic curve,
          weighted averaged
        - roc_auc_samples: area under the receiver operating characteristic curve,
          samples averaged
        - pr_auc_micro: area under the precision recall curve, micro averaged
        - pr_auc_macro: area under the precision recall curve, macro averaged
        - pr_auc_weighted: area under the precision recall curve, weighted averaged
        - pr_auc_samples: area under the precision recall curve, samples averaged
        - accuracy: accuracy score
        - f1_micro: f1 score, micro averaged
        - f1_macro: f1 score, macro averaged
        - f1_weighted: f1 score, weighted averaged
        - f1_samples: f1 score, samples averaged
        - precision_micro: precision score, micro averaged
        - precision_macro: precision score, macro averaged
        - precision_weighted: precision score, weighted averaged
        - precision_samples: precision score, samples averaged
        - recall_micro: recall score, micro averaged
        - recall_macro: recall score, macro averaged
        - recall_weighted: recall score, weighted averaged
        - recall_samples: recall score, samples averaged
        - jaccard_micro: Jaccard similarity coefficient score, micro averaged
        - jaccard_macro: Jaccard similarity coefficient score, macro averaged
        - jaccard_weighted: Jaccard similarity coefficient score, weighted averaged
        - jaccard_samples: Jaccard similarity coefficient score, samples averaged
        - ddi: drug-drug interaction score (specifically for drug-related tasks, such as drug recommendation)
        - hamming_loss: Hamming loss
        - cwECE: classwise ECE (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_classwise`.
        - cwECE_adapt: classwise adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_classwise`.

    The following metrics related to the prediction sets are accepted as well, but will be ignored if y_predset is None:
        - fp: Number of false positives.
        - tp: Number of true positives.


    If no metrics are specified, pr_auc_samples is computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples, n_labels).
        y_prob: Predicted probabilities of shape (n_samples, n_labels).
        metrics: List of metrics to compute. Default is ["pr_auc_samples"].
        threshold: Threshold to binarize the predicted probabilities. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import multilabel_metrics_fn
        >>> y_true = np.array([[0, 1, 1], [1, 0, 1]])
        >>> y_prob = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0.6]])
        >>> multilabel_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.5}
    """
    if metrics is None:
        metrics = ["pr_auc_samples"]
    prediction_set_metrics = ['tp', 'fp']

    if aux_data is None:
        aux_data = {'y_grouped':None, 'p_grouped':None,  'topk':20}
    else:
        aux_data['topk'] = 20 # add
        aux_data['y_grouped'] = None # 我们不标定这些疾病属于哪个group

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "roc_auc_micro":
            roc_auc_micro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="micro"
            )
            output["roc_auc_micro"] = roc_auc_micro
        elif metric =='avg_drug':
            output['avg_drug'] = np.mean(y_pred.sum(1))
        elif metric == "roc_auc_macro":
            roc_auc_macro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro"
            )
            output["roc_auc_macro"] = roc_auc_macro
        elif metric == "roc_auc_weighted":
            roc_auc_weighted = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted"
            )
            output["roc_auc_weighted"] = roc_auc_weighted
        elif metric == "roc_auc_samples":
            roc_auc_samples = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="samples"
            )
            output["roc_auc_samples"] = roc_auc_samples
        elif metric == "pr_auc_micro":
            pr_auc_micro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="micro"
            )
            output["pr_auc_micro"] = pr_auc_micro
        elif metric == "pr_auc_macro":
            pr_auc_macro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="macro"
            )
            output["pr_auc_macro"] = pr_auc_macro
        elif metric == "pr_auc_weighted":
            pr_auc_weighted = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="weighted"
            )
            output["pr_auc_weighted"] = pr_auc_weighted
        elif metric == "pr_auc_samples":
            pr_auc_samples = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="samples"
            )
            output["pr_auc_samples"] = pr_auc_samples
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true.flatten(), y_pred.flatten())
            output["accuracy"] = accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro") # Diag不准
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "f1_samples":
            f1_samples = sklearn_metrics.f1_score(y_true, y_pred, average="samples",  zero_division=1)
            output["f1_samples"] = f1_samples
        elif metric == "precision_micro":
            precision_micro = sklearn_metrics.precision_score(
                y_true, y_pred, average="micro"
            )
            output["precision_micro"] = precision_micro
        elif metric == "precision_macro":
            precision_macro = sklearn_metrics.precision_score(
                y_true, y_pred, average="macro"
            )
            output["precision_macro"] = precision_macro
        elif metric == "precision_weighted":
            precision_weighted = sklearn_metrics.precision_score(
                y_true, y_pred, average="weighted"
            )
            output["precision_weighted"] = precision_weighted
        elif metric == "precision_samples":
            precision_samples = sklearn_metrics.precision_score(
                y_true, y_pred, average="samples"
            )
            output["precision_samples"] = precision_samples
        elif metric == "recall_micro":
            recall_micro = sklearn_metrics.recall_score(y_true, y_pred, average="micro")
            output["recall_micro"] = recall_micro
        elif metric == "recall_macro":
            recall_macro = sklearn_metrics.recall_score(y_true, y_pred, average="macro")
            output["recall_macro"] = recall_macro
        elif metric == "recall_weighted":
            recall_weighted = sklearn_metrics.recall_score(
                y_true, y_pred, average="weighted"
            )
            output["recall_weighted"] = recall_weighted
        elif metric == "recall_samples":
            recall_samples = sklearn_metrics.recall_score(
                y_true, y_pred, average="samples"
            )
            output["recall_samples"] = recall_samples
        elif metric == "jaccard_micro":
            jaccard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jaccard_micro
        elif metric == "jaccard_macro":
            jaccard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jaccard_macro
        elif metric == "jaccard_weighted":
            jaccard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jaccard_weighted
        elif metric == "jaccard_samples":
            jaccard_samples = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="samples",  zero_division=1
            )
            output["jaccard_samples"] = jaccard_samples
        elif metric == "hamming_loss":
            hamming_loss = sklearn_metrics.hamming_loss(y_true, y_pred)
            output["hamming_loss"] = hamming_loss
        elif metric == "ddi":
            ddi_adj = np.load(os.path.join(CACHE_PATH, 'ddi_adj.npy'))
            y_pred = [np.where(item)[0] for item in y_pred]
            output["ddi_score"] = ddi_rate_score(y_pred, ddi_adj)
        elif metric == "topk_acc": # 这里的topk和rec还是有点不同
             all_acc, all_k_acc = topk_acc(y_true, y_prob, k=aux_data['topk'], grouped_y=aux_data['y_grouped'])
             output['topk_acc'] = all_acc
             output['topk_acc_grouped'] = all_k_acc
        elif metric == "topk_precision":
            visit_precision = topk_precision(y_true, y_prob, k=aux_data['topk'])
            # all_prec, all_k_prec = topk_precision_group(y_true, y_prob, k=aux_data['topk'], grouped_y=aux_data['y_grouped'])
            output['topk_precision'] = visit_precision # 这里有点奇怪啊
            # output['topk_precision_grouped'] = all_k_prec
            # output['topk_prec'] = all_prec

        elif metric == "group_rec":
            output['rec_grouped'] = group_rec(y_true, y_prob, y_pred, patient_ids, group_p=aux_data['p_grouped'])

        elif metric in {"cwECE", "cwECE_adapt"}:
            output[metric] = calib.ece_classwise(
                y_prob,
                y_true,
                bins=20,
                adaptive=metric.endswith("_adapt"),
                threshold=0.0,
            )
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == 'tp':
                output[metric] = (y_true * y_predset).sum(1).mean()
            elif metric == 'fp':
                output[metric] = ((1-y_true) * y_predset).sum(1).mean()
        else:
            raise ValueError(f"Unknown metric for multilabel classification: {metric}")

    return output






def binary_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.5,
        aux_data: Optional[Dict[str, Any]] = None,
        patient_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Computes metrics for binary classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - pr_auc: area under the precision-recall curve
        - roc_auc: area under the receiver operating characteristic curve
        - accuracy: accuracy score
        - balanced_accuracy: balanced accuracy score (usually used for imbalanced
          datasets)
        - f1: f1 score
        - precision: precision score
        - recall: recall score
        - cohen_kappa: Cohen's kappa score
        - jaccard: Jaccard similarity coefficient score
        - ECE: Expected Calibration Error (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_binary`.
        - ECE_adapt: adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_binary`.
    If no metrics are specified, pr_auc, roc_auc and f1 are computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        metrics: List of metrics to compute. Default is ["pr_auc", "roc_auc", "f1"].
        threshold: Threshold for binary classification. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
    """
    if metrics is None:
        metrics = ["pr_auc", "roc_auc", "f1"]

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "pr_auc":
            pr_auc = sklearn_metrics.average_precision_score(y_true, y_prob)
            output["pr_auc"] = pr_auc
        elif metric == "roc_auc":
            roc_auc = sklearn_metrics.roc_auc_score(y_true, y_prob)
            output["roc_auc"] = roc_auc
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1":
            f1 = sklearn_metrics.f1_score(y_true, y_pred)
            output["f1"] = f1
        elif metric == "precision":
            precision = sklearn_metrics.precision_score(y_true, y_pred)
            output["precision"] = precision
        elif metric == "recall":
            recall = sklearn_metrics.recall_score(y_true, y_pred)
            output["recall"] = recall
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "jaccard":
            jaccard = sklearn_metrics.jaccard_score(y_true, y_pred)
            output["jaccard"] = jaccard
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = calib.ece_confidence_binary(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
            )
        else:
            raise ValueError(f"Unknown metric for binary classification: {metric}")
    return output

