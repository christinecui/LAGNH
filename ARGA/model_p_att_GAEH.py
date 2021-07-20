import torch
import torch.nn as nn
import torch.nn.functional as F
from ARGA.layers import (GraphConvolution, gaussian_noise_layer, Self_Attn)
import utils.utils as utils
class ARGA(nn.Module):
    '''
            Graph Auto-Encoders.
    '''

    def __init__(self, input_feat_dim=4096, hidden_dim1=2048, hidden_dim2=32, dropout=0.5, n_class=80):
        super(ARGA, self).__init__()
        self.gc1 = GraphConvolution(512, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, lambda x: x)

        # ADD in 2020-11-24
        self.classify = nn.Linear(hidden_dim2, n_class)
        self.hash = nn.Linear(hidden_dim2, hidden_dim2)

        # ADD in 2021-3-10
        self.common_img = nn.Linear(input_feat_dim, 512)
        self.common_pl = nn.Linear(n_class, 512)

        self.common_cl_x = nn.Linear(512, n_class)
        self.common_pl_x = nn.Linear(512, n_class)

        self.attention_plabel = nn.Sequential(
            nn.Linear(n_class, n_class),
            nn.Softmax()
        )

        self.s_attention = Self_Attn()

    def encoder(self, x, adj):
        hidden1 = self.gc1(x, adj)
        output = self.gc2(hidden1, adj)
        return output

    def forward(self, x, adj, plabel, gamma, fusion_factor):
        # Add in 2021-3-10
        common_x = self.common_img(x)
        common_p = self.common_pl(plabel)

        _, out_aff, _ = utils.affinity_cos_gpu(common_x, common_p, flag=False)
        att_x = torch.matmul(out_aff, common_p)
        fusion_x = torch.add(common_x, att_x)

        p_feat = plabel 
        _, adj_a, adj_p, adj_f = utils.affinity_fusion_gpu_GAEH_103(fusion_x, p_feat, flag=False, fusion_factor=fusion_factor)

        z = self.encoder(fusion_x, adj_a)
        c = self.classify(z) 

        return self.dc(z, gamma), z, c, self.common_cl_x(common_x), self.common_pl_x(common_p), p_feat, adj_a, adj_p, adj_f

class InnerProductDecoder(nn.Module):
    '''
        Decoder in GAE or VAGE.
    '''
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, gamma):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z, z.t()) * gamma
        return adj


class Discriminator(nn.Module):

    def __init__(self, n_bits=32):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(n_bits, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.discriminator(x)
