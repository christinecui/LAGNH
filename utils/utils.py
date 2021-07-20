import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import sklearn.preprocessing as pp
import torch


def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))
    out_affnty = affnty/col_sum # row data sum = 1
    in_affnty = np.transpose(affnty/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

def L21_norm(X):
    # 2-norm for column
    X_norm2 = np.linalg.norm(X, ord=2, axis=0)
    X_norm2_norm1 = np.linalg.norm(X_norm2, ord=1)
    return X_norm2_norm1

def kernelize(data, anchor):

    _, _, aff = graph_eculi(data, anchor)
    gamma = np.sum(np.sum(aff, axis=1)) / (aff.shape[0] * aff.shape[1])
    gamma2 = 2 * gamma * gamma
    aff = (aff ** 2) / gamma2
    aff_exp = np.exp(aff)
    return aff_exp

############################################affinity#############################################

# Check in 2020-6-25(14:26)
def getTrue2(test_label: np.ndarray, train_label: np.ndarray):
    cateTestTrain = np.sign(np.matmul(test_label, train_label.T)) # 0 or 1
    cateTestTrain = cateTestTrain.astype('int16')
    return cateTestTrain

def affinity_eculi_gpu(data_1: torch.Tensor, data_2: torch.Tensor, I_size=0, theta=2, topk=0):
    XYt = torch.matmul(data_1, data_2.T)
    X2, Y2 = torch.mul(data_1, data_1), torch.mul(data_2, data_2)
    X2_sum = torch.sum(X2, dim=1).reshape(X2.size()[0], 1)
    Y2_sum = torch.sum(Y2, dim=1).reshape(1, Y2.size()[0])
    tmp = X2_sum + Y2_sum - 2 * XYt
    tmp[tmp < 0] = 0
    affinity = tmp / theta
    new_affinity = torch.exp(-affinity)
    if I_size != 0:
        I = Variable(torch.eye(I_size)).cuda()
        new_affinity = torch.cat([I, new_affinity], dim=1)

    in_aff = torch.nn.functional.normalize(new_affinity, p=2, dim=0).T
    out_aff = torch.nn.functional.normalize(new_affinity, p=2, dim=1)

    return in_aff, out_aff, new_affinity

def affinity_fusion(feature: np.ndarray, plabel: np.ndarray, flag=True, fusion_factor=0):
    '''

    :param feature:
    :param plabel:
    :param flag: "flag=True" means that feature will be normalized.
    :param fusion_factor: the factor of feature cosine-similarity
    :return:
    '''
    if flag: # flag==false
        pro_feature = pp.normalize(feature, norm='l2') # to calculate the cos-similarity
    else:
        pro_feature = feature

    _, aff_norm_f, aff_label_f = affinity_eculi(feature, feature)
    _, aff_norm_p, aff_label_p = affinity_tag_multi(plabel, plabel)

    in_aff, out_aff = normalize(aff_label_f * fusion_factor + aff_label_p) # feature graph * factor + plabel graph
    return in_aff, out_aff, aff_label_p, aff_label_f

def affinity_fusion_gpu_GAEH_103(feature: torch.Tensor, plabel: torch.Tensor, flag=True, fusion_factor=0):
    if flag:
        pro_feature = torch.nn.functional.normalize(feature)
    else:
        pro_feature = feature

    _, aff_norm_f, aff_label_f = affinity_eculi_gpu(feature, feature)
    _, aff_norm_p, aff_label_p = affinity_cos_gpu(plabel, plabel)

    new_affinity = aff_label_f  + aff_label_p * fusion_factor  # GAEH_103 change

    in_aff = torch.nn.functional.normalize(new_affinity, p=1, dim=0).T
    out_aff = torch.nn.functional.normalize(new_affinity, p=1, dim=1)

    return in_aff, out_aff, aff_label_p, aff_label_f

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix[affinity_matrix > 1] = 1
    in_aff, out_aff = normalize(affinity_matrix)
    return in_aff, out_aff, affinity_matrix

def affinity_eculi(data_1: np.ndarray, data_2: np.ndarray)->np.ndarray:

    XYt = np.matmul(data_1, data_2.T)
    X2, Y2 = data_1 ** 2, data_2 ** 2
    X2_sum = np.sum(X2, axis=1).reshape(X2.shape[0], 1)
    Y2_sum = np.sum(Y2, axis=1).reshape(1, Y2.shape[0])
    tmp = X2_sum + Y2_sum - 2 * XYt
    tmp[tmp < 0] = 0 
    affinity = np.sqrt(tmp)
    affinity = np.exp(-affinity)

    in_aff, out_aff = normalize(affinity)
    return in_aff, out_aff, affinity

def affinity_cos_gpu(data_1: torch.Tensor, data_2: torch.Tensor, flag=True, I_size=0):

    if not flag: # Flag=False
        pro_data_1 = torch.nn.functional.normalize(data_1, p=2, dim=1)  # to calculate the cos-similarity
        pro_data_2 = torch.nn.functional.normalize(data_2, p=2, dim=1)
    else:
        pro_data_1 = data_1
        pro_data_2 = data_2

    affinity = torch.matmul(pro_data_1, pro_data_2.T)
    affinity[affinity < 0] = 0
    if I_size != 0:
        I = Variable(torch.eye(I_size)).cuda()
        affinity = torch.cat([I, affinity], dim=1)

    in_aff = torch.nn.functional.normalize(affinity, p=2, dim=0).T
    out_aff = torch.nn.functional.normalize(affinity, p=2, dim=1)
    return in_aff, out_aff, affinity

import random
import os
import numpy as np
import torch
def seed_setting(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True