import torch.utils.data as data
import torch
import numpy as np
import pickle as pkl
from collections import namedtuple

## COCO-data
paths_COCO = {
    'COCO_train': '/Data/data/COCO/pkl/train2017.pkl',
    'COCO_val': '/Data/data/COCO/pkl/val2017.pkl',
    'COCO_retrieval': '/Data/data/COCO/pkl/retrieval2017.pkl',
}
## NUSWIDE
paths_NUSWIDE = {
    'NUSWIDE_train': '/Data/data/NUSWIDE/pkl/train.pkl',
    'NUSWIDE_val': '/Data/data/NUSWIDE/pkl/val.pkl',
    'NUSWIDE_retrieval': '/Data/data/NUSWIDE/pkl/retrieval.pkl'
}

dataset_lite = namedtuple('dataset_lite', ['feature', 'label', 'plabel'])

def load_coco(n_bits, mode):
    if mode == 'train':
        data = pkl.load(open(paths_COCO['COCO_train'], 'rb'))
        feature = data['train_feature']
        label = data['train_label']
        plabel = data['train_tag']

    elif mode == 'retrieval':
        data = pkl.load(open(paths_COCO['COCO_retrieval'], 'rb'))
        feature = data['retrieval_feature']
        label = data['retrieval_label']
        plabel = data['retrieval_tag']

    else:
        data = pkl.load(open(paths_COCO['COCO_val'], 'rb'))
        feature = data['val_feature']
        label = data['val_label']
        plabel = data['val_tag']

    return dataset_lite(feature, label, plabel)

def load_nuswide(n_bits, mode):
    if mode == 'train':
        data = pkl.load(open(paths_NUSWIDE['NUSWIDE_train'], 'rb'))
        feature = data['train_feature']
        label = data['train_label']
        plabel = data['train_tag']

    elif mode == 'retrieval':
        data = pkl.load(open(paths_NUSWIDE['NUSWIDE_retrieval'], 'rb'))
        feature = data['retrieval_feature']
        label = data['retrieval_label']
        plabel = data['retrieval_tag']

    else:
        data = pkl.load(open(paths_NUSWIDE['NUSWIDE_val'], 'rb'))
        feature = data['val_feature']
        label = data['val_label']
        plabel = data['val_tag']

    return dataset_lite(feature, label, plabel)

class my_dataset(data.Dataset):
    def __init__(self, feature, plabel):
        self.feature = torch.Tensor(feature)
        self.plabel = torch.Tensor(plabel)
        self.length = self.feature.size(0)

    def __getitem__(self, item):
        return self.feature[item, :], self.plabel[item, :]

    def __len__(self):
        return self.length

