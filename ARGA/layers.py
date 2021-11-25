import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    '''
        GCN layer.
    '''
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.mm(input, self.weight)
        output = torch.mm(adj, output)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gaussian_noise_layer(input, std):
    noise = torch.normal(std=std, size=input.size())
    return input + noise

def similarity_loss(theta: torch.Tensor, S: torch.Tensor):
    loss = - S.mul(theta) + torch.log(1 + torch.exp(theta))
    return loss

class Self_Attn(nn.Module):
    def __init__(self):
        super(Self_Attn, self).__init__()

        self.query_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.key_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.value_conv = nn.Conv2d(1, 1, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        b_size, ndim = x.size()
        U_x = torch.unsqueeze(x, dim=1).float()
        U_x = torch.unsqueeze(U_x, dim=1).float()

        proj_query = self.query_conv(U_x).view(b_size, -1, ndim).permute(0, 2, 1)
        proj_key = self.key_conv(U_x).view(b_size, -1, ndim)
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = self.value_conv(U_x).view(b_size, -1, ndim)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = torch.squeeze(out)
        out = self.gamma * out + x

        return out, attention