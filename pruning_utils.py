# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : pruning_utils.py
@Project  : EarlyRobust
@Time     : 2022/10/12 12:17
@Author   : Zhiheng Xi
"""
import numpy as np
import torch
from transformers.models.bert.modeling_bert import BertLayer

if __name__ == '__main__':
    pass

def cal_mask_distance_in_self_heads_of_two_step_prune_heads_global(model, self_slimming_coef_records,
                                                slimming_step1, slimming_step2,
                                                self_pruning_method, self_pruning_ratio
                                                ):
    """

    @param model:
    @param self_slimming_coef_records:
    @param slimming_step1:
    @param slimming_step2:
    @param self_pruning_method:
    @param self_pruning_ratio:
    @return:
    """
    slimming_coefs1 = self_slimming_coef_records[:, slimming_step1, :]
    slimming_coefs2 = self_slimming_coef_records[:, slimming_step2, :]


    def is_every_layer_one_head_survived(slimming_coefs,quantile_axis,threshold):
        layers_masks = slimming_coefs > threshold
        for idx in range(len(layers_masks)):
            mask = layers_masks[idx]
            if sum([1 if i==True else 0 for i in mask ])==0:
                return idx
        return -1

    # cal layer_mask_1
    quantile_axis1 = -1 if self_pruning_method == 'layerwise' else None
    threshold1 = np.quantile(slimming_coefs1, self_pruning_ratio, axis=quantile_axis1,keepdims=True)  # 找到对应的分位数

    while True:
        idx = is_every_layer_one_head_survived(slimming_coefs1,quantile_axis1,threshold1)
        if idx != -1:
            p = list(slimming_coefs1[idx]).index(max(slimming_coefs1[idx]))
            slimming_coefs1[idx][p]= float('inf')
            quantile_axis1 = -1 if self_pruning_method == 'layerwise' else None
            threshold1 = np.quantile(slimming_coefs1, self_pruning_ratio, axis=quantile_axis1, keepdims=True)  # 找到对应的分位数
        else:
            break

    layers_masks1 = slimming_coefs1 > threshold1

    # cal layer_mask_2
    quantile_axis2 = -1 if self_pruning_method == 'layerwise' else None
    threshold2 = np.quantile(slimming_coefs2, self_pruning_ratio, axis=quantile_axis2,keepdims=True)  # 找到对应的分位数

    while True:
        idx = is_every_layer_one_head_survived(slimming_coefs2,quantile_axis2,threshold2)
        if idx != -1:
            p = list(slimming_coefs2[idx]).index(max(slimming_coefs2[idx]))
            slimming_coefs2[idx][p]= float('inf')
            quantile_axis2 = -1 if self_pruning_method == 'layerwise' else None
            threshold2 = np.quantile(slimming_coefs2, self_pruning_ratio, axis=quantile_axis2, keepdims=True)  # 找到对应的分位数
        else:
            break

    layers_masks2 = slimming_coefs2 > threshold2

    # cal distance
    argwhere = np.argwhere(layers_masks1 != layers_masks2)
    distance = len(argwhere)

    normalized_dis1 = torch.tensor(distance/np.size(layers_masks1))
    normalized_dis2 = 1.0-torch.cosine_similarity(torch.tensor(layers_masks1.astype(float)).reshape(1,-1),torch.tensor(layers_masks2.astype(float)).reshape(1,-1))
    # print(normalized_dis1==normalized_dis2)
    return distance,normalized_dis1


def cal_mask_distance_in_inter_neurons_of_two_step(model,inter_slimming_coef_records,
                                                      slimming_step1, slimming_step2,
                                                      inter_pruning_method,inter_pruning_ratio
                                                      ):
    """
    @param model:
    @param inter_slimming_coef_records:
    @param slimming_step1:
    @param slimming_step2:
    @param inter_pruning_method:
    @param inter_pruning_ratio:
    @return:
    """
    slimming_coefs1 = inter_slimming_coef_records[:, slimming_step1, :]
    slimming_coefs2 = inter_slimming_coef_records[:, slimming_step2, :]

    # slimming_coefs1 = slimming_coefs1[0]
    # slimming_coefs2 = slimming_coefs2[0]

    bert_layers = []
    for m in model.modules():
        if isinstance(m, BertLayer):
            bert_layers.append(m)
    quantile_axis = -1 if inter_pruning_method == 'layerwise' else None
    threshold1 = np.quantile(slimming_coefs1, inter_pruning_ratio, axis=quantile_axis, keepdims=True)
    threshold2 = np.quantile(slimming_coefs2, inter_pruning_ratio, axis=quantile_axis, keepdims=True)

    layers_masks1 = slimming_coefs1 > threshold1
    layers_masks2 = slimming_coefs2 > threshold2
    argwhere = np.argwhere(layers_masks1 != layers_masks2)
    distance = len(argwhere)
    normalized_dis1 = torch.tensor(distance/np.size(layers_masks1))
    normalized_dis2 = 1.0-torch.cosine_similarity(torch.tensor(layers_masks1.astype(float)).reshape(1,-1),torch.tensor(layers_masks2.astype(float)).reshape(1,-1))
    # print(normalized_dis1==normalized_dis2)
    return distance,normalized_dis1

