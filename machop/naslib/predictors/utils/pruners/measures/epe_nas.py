# MIT License

# Copyright (c) 2021 VascoLopes

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Implementation of EPE-NAS: Efficient Performance Estimation Without Training
# for Neural Architecture Search (https://arxiv.org/abs/2102.08099) taken from the
# authors' Github repository https://github.com/VascoLopes/EPE-NAS/blob/main/search.py

import torch
import numpy as np
import pdb
from . import measure


def compute_epe_score2(net, inputs, targets, loss_fn, split_data=1):
    net.zero_grad()

    inputs.requires_grad_(True)
    outputs = net(inputs)  # 在移动 inputs 到设备之前定义 outputs
    inputs = inputs.to(outputs.device)  # 将 inputs 张量移动到与 outputs 张量相同的设备上
    targets = targets.to(outputs.device)
    loss = loss_fn(outputs, targets)
    loss.backward()
    jacobian = inputs.grad.detach().cpu()  # 获取雅可比矩阵，并转换为 CPU 上的张量
    inputs=inputs.to("cpu")
    targets=targets.to("cpu")
    # 计算每个类别的雅可比矩阵的相关系数矩阵
    corr_matrices = {}
    unique_labels = torch.unique(targets)
    for label in unique_labels:
        label_indices = torch.where(targets == label)[0].to("cpu")  # 将索引移动到与 outputs 张量相同的设备上
        jacobian_label = jacobian[label_indices]
        corr_matrix = torch.cov(jacobian_label.t())
        corr_matrices[label.item()] = corr_matrix
    
    # 计算性能评分
    score = 0
    for corr_matrix in corr_matrices.values():
        score += torch.sum(torch.abs(corr_matrix))
    # score /= len(corr_matrices)  # 取平均值作为最终评分
    
    return score.item()
# def get_batch_jacobian(net, x, target, to, device, args=None):
#     net.zero_grad()
#     x.requires_grad_(True)
#     y = net(x)
#     y.backward(torch.ones_like(y))
#     jacob = x.grad.detach()
#     return jacob, target.detach(), y.shape[-1]

# def eval_score_perclass(jacob, labels=None, n_classes=10):
#     k = 1e-5
    
#     per_class={}
#     for i, label in enumerate(labels[0]):
#         if label in per_class:
#             per_class[label] = np.vstack((per_class[label],jacob[i]))
#         else:
#             per_class[label] = jacob[i]

    
#     ind_corr_matrix_score = {}
#     for c in per_class.keys():
#         s = 0
#         # try:
#         corrs = np.array(np.corrcoef(per_class[c]))
#         if c == 2:
#             print(corrs)
#             s = np.sum(np.log(abs(corrs)+k))
            
        
#         s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
#         if n_classes > 100:
#             s /= len(corrs)
#         # except: # defensive programming
#         #     continue
#         ind_corr_matrix_score[c] = s
#     # per class-corr matrix A and B
#     score = 0
#     ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()

#     if n_classes <= 100:
#         for c in ind_corr_matrix_score_keys:
#             # B)
#             score += np.absolute(ind_corr_matrix_score[c])
#     else:
#         for c in ind_corr_matrix_score_keys:
#             # A)
#             for cj in ind_corr_matrix_score_keys:
#                 score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

#         if len(ind_corr_matrix_score_keys) > 0:
#             # should divide by number of classes seen
#             score /= len(ind_corr_matrix_score_keys)

#     return score


# @measure("epe_nas")
# def compute_epe_score(net, inputs, targets, loss_fn, split_data=1):
#     jacobs = []
#     labels = []

#     # try:
#     jacobs_batch, target, n_classes = get_batch_jacobian(net, inputs, targets, None, None)
#     jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
#     if len(target.shape) == 2: # Hack to handle TNB101 classification tasks
#         target = torch.argmax(target, dim=1)
#     labels.append(target.cpu().numpy())
#     jacobs = np.concatenate(jacobs, axis=0)

#     s = eval_score_perclass(jacobs, labels, n_classes)

#     # except Exception as e:
#     #     print(e)
#     #     s = np.nan

#     return s

def get_batch_jacobian(net, x, target, to, device, args=None):
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach(), y.shape[-1]

def eval_score_perclass(jacob, labels=None, n_classes=10):
    k = 1e-5

    per_class={}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label],jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.array(np.corrcoef(per_class[c]))

            s = np.sum(np.log(abs(corrs)+k))#/len(corrs)
            if n_classes > 100:
                s /= len(corrs)
        except: # defensive programming
            continue
        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:

        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else:
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c]-ind_corr_matrix_score[cj])

        if len(ind_corr_matrix_score_keys) > 0:
            # should divide by number of classes seen
            score /= len(ind_corr_matrix_score_keys)

    return score


@measure("epe_nas")
def compute_epe_score(net, inputs, targets, loss_fn, split_data=1):
    jacobs = []
    labels = []

    try:

        jacobs_batch, target, n_classes = get_batch_jacobian(net, inputs, targets, None, None)
        jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())

        if len(target.shape) == 2: # Hack to handle TNB101 classification tasks
            target = torch.argmax(target, dim=1)

        labels.append(target.cpu().numpy())

        jacobs = np.concatenate(jacobs, axis=0)

        s = eval_score_perclass(jacobs, labels, n_classes)

    except Exception as e:
        print(e)
        s = compute_epe_score2(net, inputs, targets, loss_fn, split_data=1)

    return s





