import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd




def sink(M, reg, numItermax=1000, stopThr=1e-9):

    # we assume that no distances are null except those of the diagonal of
    # distances

    a = Variable(torch.ones((M.size()[0],)) / M.size()[0])
    b = Variable(torch.ones((M.size()[1],)) / M.size()[1])

    # init data
    Nini = len(a)
    Nfin = len(b)

    u = Variable(torch.ones(Nini) / Nini)
    v = Variable(torch.ones(Nfin) / Nfin)

    # print(reg)

    K = torch.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        #print(T(K).size(), u.view(u.size()[0],1).size())
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1. / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2).data[0]


        cpt += 1

    return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)

def pairwise_distances(x, y):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)

    return dist.float()

def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

    return mm


def uniform_example(batch_size = 100):
    m_list = ((np.array(list(range(1, 10))) / 5.0 - 1)).tolist()
    for mu in m_list:
        x = np.zeros((batch_size, 2))
        y = np.zeros((batch_size, 2))
        x[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 1] = np.random.uniform(0, 1, batch_size)

        y[:, 0] = mu

        #x = [[1,2], [3,2]]
        #y = [[2,1], [4,1]]
        # M = [[0., 1.], [1., 0.]]

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # M = np.asarray(M, dtype=np.float64)


        x = Variable(torch.from_numpy(x).float())
        y = Variable(torch.from_numpy(y).float())

        #M = torch.pow(x - y, 2).sum(1)
        M = pairwise_distances(x,y)
        print(sink(M, reg=1))
        #return sink(M,1)


if __name__ == '__main__':
    uniform_example()

