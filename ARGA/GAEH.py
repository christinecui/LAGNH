import argparse
import time
import sys
sys.path.append('..')

import scipy.io as sio

from utils.datasets import *
from utils.utils import *
from ARGA.model_p_att_GAEH import ARGA, Discriminator

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ARGA', help='Use ARGA, ARVGA.')
parser.add_argument('--dataset', type=str, default='COCO')
parser.add_argument('--nbits', type=int, default=64)
parser.add_argument('--n_class', type=int, default=80)

parser.add_argument('--input_dim', type=int, default=4096)
parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in GCN hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--GD_lr', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.5, help='')

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--fusion_factor', type=float, default=10)
parser.add_argument('--alpha', type=float, default=0.1, help='The factor of classification loss.')
parser.add_argument('--beta', type=float, default=10, help='kS-cos(B, B) from DJSRH')
parser.add_argument('--lamda', type=float, default=1, help='z-b loss.')
parser.add_argument('--cor', type=float, default=0, help='corr param')
parser.add_argument('--bal', type=float, default=0, help='balance param')
parser.add_argument('--k', type=float, default=1, help='scale factor of origin S')
parser.add_argument('--gamma', type=float, default=1, help='Make zz^T too large.')

parser.add_argument('--mu', type=float, default=0, help='Simlarity loss by negetive likehood.(Laplace matrix, tr(BLB))')
parser.add_argument('--theta', type=float, default=10, help='GAE reconstruction loss.')

args = parser.parse_args()

def train_ARGA(args):
    args.fusion_factor = 0

    Epochs = args.epochs
    nbits = args.nbits
    hidden1 = args.hidden1
    hidden2 = args.hidden2
    input_dim = args.input_dim

    if hidden2 != nbits:
        print('The GCN-2 layer output dim is not equal to Hashcode length.')
        hidden2 = nbits

    # 1. load data
    dset = 0
    if args.dataset == 'COCO':
        dset = load_coco(nbits, mode='train')
    elif args.dataset == 'NUSWIDE':
        dset = load_nuswide(nbits, mode='train')
    else:
        print('No data can be used!')
    print('Load %s dataset.'%(args.dataset))

    n_nodes = args.batch_size
    train_loader = data.DataLoader(my_dataset(dset.feature, dset.plabel),
                                   batch_size=args.batch_size,
                                   shuffle=True)

    arga = ARGA(input_feat_dim=input_dim,
                hidden_dim1=hidden1,
                hidden_dim2=hidden2,
                dropout=args.dropout, n_class=args.n_class)
    discriminator = Discriminator(n_bits=nbits)

    arga.cuda()
    discriminator.cuda()

    # 3. loss
    loss_adv = torch.nn.BCEWithLogitsLoss()
    loss_l2 = torch.nn.MSELoss()
    loss_cl = torch.nn.MultiLabelSoftMarginLoss()

    # 4. opt
    optimizer_G = torch.optim.Adam(arga.parameters(), lr=args.GD_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.GD_lr)
    optimizer = torch.optim.Adam(arga.parameters(), lr=args.lr)

    start_time = time.time() * 1000
    LOSS = []
    for epoch in range(Epochs):
        for i, (feature, plabel) in enumerate(train_loader):
            _, aff_norm, aff_label_p, aff_label_f = affinity_fusion(feature.numpy(),
                                                     plabel.numpy(),
                                                     flag=False,
                                                     fusion_factor=args.fusion_factor)

            pos_weight = float(aff_norm.shape[0] * aff_norm.shape[0] - aff_norm.sum()) / aff_norm.sum()
            pos_weight = torch.as_tensor(pos_weight)
            norm = 1

            aff_norm = Variable(torch.Tensor(aff_norm)).cuda()
            aff_label_p = Variable(torch.Tensor(aff_label_p).cuda())
            feature = Variable(feature).cuda()
            plabel = Variable(plabel).cuda()
            pos_weight = Variable(pos_weight).cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            adj_cons, z, c, cl_x, cl_p, p_feat, adj_a, adj_p, adj_f = arga(feature, aff_norm, plabel, args.gamma, args.fusion_factor) # use normalize affinity to create GCN graph

            optimizer_D.zero_grad()

            d_fake = discriminator(z)
            real_contrib = Variable(torch.randn(size=(z.size()[0], z.size()[1]))).cuda() # guass
            d_real = discriminator(real_contrib)

            dc_loss_real = loss_adv(d_real, Variable(torch.ones(d_real.shape[0], 1)).cuda())
            dc_loss_fake = loss_adv(d_fake, Variable(torch.zeros(d_fake.shape[0], 1)).cuda())
            loss_D = dc_loss_fake + dc_loss_real

            loss_D.backward()
            optimizer_D.step()
            if i + 1 == len(train_loader) and (epoch+1) % 5 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f, Loss-real: %.4f, Loss-fake: %.4f'
                    % (epoch + 1, Epochs, loss_D.item(), dc_loss_real.item(), dc_loss_fake.item()))

            # -----------------
            #  Train Generator
            # -----------------
            for j in range(5):
                optimizer_G.zero_grad()

                adj_cons, z, c, cl_x, cl_p, p_feat, adj_a, adj_p, adj_f = arga(feature, aff_norm, plabel, args.gamma, args.fusion_factor)
                z_norm = F.normalize(z)
                B = torch.sign(z) # -1 / 1
                d_fake = discriminator(z)

                # loss1
                generator_loss = loss_adv(d_fake, Variable(torch.ones(d_fake.shape[0], 1)).cuda())

                # loss2
                sign_loss = loss_l2(z, B) * args.lamda

                # loss3
                cl_loss = loss_cl(c, p_feat)  * args.alpha

                # loss4
                balance_loss = torch.sum(z) / z.size(0) * args.bal
                cor_loss = torch.norm(z.t().mm(z), 2) * args.cor

                # loss 5
                temp = z_norm.mm(z_norm.t())
                temp[temp < 0] = 0
                construct_loss = loss_l2(temp, args.k * adj_p) * args.beta

                # total loss
                loss_G = generator_loss + sign_loss + cl_loss + construct_loss + cor_loss + balance_loss

                LOSS.append(loss_G)

                loss_G.backward()
                optimizer_G.step()
                if i + 1 == len(train_loader) and (epoch + 1) % 5 == 0 and j == 4:
                    print('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-B: %.4f, Loss-G: %.4f'
                          % (epoch + 1, Epochs, loss_G.item(), construct_loss.item(), sign_loss.item(), generator_loss.item()))

    end_time = time.time() * 1000
    train_time = end_time - start_time
    print('[Train time] %.4f' % (train_time / 1000))
    return arga, LOSS

def test(model, args):
    model.eval()
    nbits = args.nbits

    ## Retrieval
    dset = 0
    if args.dataset == 'COCO':
        dset = load_coco(nbits, mode='retrieval')
    elif args.dataset == 'NUSWIDE':
        dset = load_nuswide(nbits, mode='retrieval')
    else:
        print('No data can be used!')

    retrieval_loader = data.DataLoader(my_dataset(dset.feature, dset.plabel),
                                       batch_size=256,
                                       shuffle=False,
                                       num_workers=0)
    retrievalP = []
    retrieval_label = dset.label
    start_time = time.time() * 1000
    for i, (feature, plabel) in enumerate(retrieval_loader):

        _, aff_norm, aff_label_p, aff_label_f = affinity_fusion(feature.numpy(), plabel.numpy(), flag=False, fusion_factor=args.fusion_factor)

        aff_label_p = Variable(torch.Tensor(aff_label_p)).cuda()
        aff_norm = Variable(torch.Tensor(aff_norm)).cuda()
        feature = Variable(feature).cuda()
        plabel = Variable(plabel).cuda()

        adj_cons, z, c, cl_x, cl_p, p_feat, adj_a, adj_p, adj_f = model(feature, aff_norm, plabel, args.gamma, args.fusion_factor)
        retrievalP.append(z.data.cpu().numpy())

    retrievalH = np.concatenate(retrievalP)
    retrievalCode = np.sign(retrievalH)
    end_time = time.time() * 1000
    retrieval_time = end_time - start_time

    ## Query
    dset = 0
    if args.dataset == 'COCO':
        dset = load_coco(nbits, mode='val')
    elif args.dataset == 'NUSWIDE':
        dset = load_nuswide(nbits, mode='val')
    else:
        print('No data can be used!')

    val_loader = data.DataLoader(my_dataset(dset.feature, dset.plabel),
                                 batch_size=256,
                                 shuffle=False,
                                 num_workers=0)
    valP = []
    val_label = dset.label
    start_time = time.time() * 1000
    for i, (feature, plabel) in enumerate(val_loader):

        _, aff_norm, aff_label_p, aff_label_f = affinity_fusion(feature.numpy(), plabel.numpy(), flag=False, fusion_factor=args.fusion_factor)

        aff_label_p = Variable(torch.Tensor(aff_label_p)).cuda()
        aff_norm = Variable(torch.Tensor(aff_norm)).cuda()
        feature = Variable(feature).cuda()
        plabel = Variable(plabel).cuda()

        adj_cons, z, c, cl_x, cl_p, p_feat, adj_a, adj_p, adj_f = model(feature, aff_norm, plabel, args.gamma, args.fusion_factor)
        valP.append(z.data.cpu().numpy())
    valH = np.concatenate(valP)
    valCode = np.sign(valH)
    end_time = time.time() * 1000
    query_time = end_time - start_time
    print('[Retrieval time] %.4f, [Query time] %.4f, [Encoding time] %.4f' % (retrieval_time / 1000, query_time / 1000, (retrieval_time + query_time) / 1000))

    ## Save
    _dict = {
        'retrieval_B': retrievalCode,
        'val_B': valCode
    }
    sava_path = 'hashcode/GCNH_' + args.dataset + '_' + str(args.nbits) + 'bits.mat'
    sio.savemat(sava_path, _dict)
    return 0

if __name__ == '__main__':
        model, LOSS = train_ARGA(args)
        test(model, args)