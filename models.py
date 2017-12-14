import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from loss_fn import loss_fn
from torch.nn import functional as F
from emd import emd
import numpy as np
import pandas as pd

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.image_dim = 28 # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.batch_size = 50

        self.discrim = nn.Sequential(
            nn.Linear(2, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 1))
        #self.mu = nn.Linear(4*self.latent_dim, 28*28)
        #self.sigma = nn.Linear(4 * self.latent_dim, 28 * 28)

        #z = Variable(torch.randn(self.batch_size, self.latent_dim))

    def forward(self, x):
        fx = self.discrim(x)

        return fx

def loss_fn(fx, fy):
    return -(torch.mean(fx) - torch.mean(fy))

def algo_dist(x,y):
    return emd(x,y)

def train(num_epochs = 1000, num_batches=50, batch_size = 100, learning_rate = 1e-5, mu=1):

    model = Critic()
    model.batch_size = batch_size

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    prev_loss = 0
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            #x = Variable(torch.randn(batch_size, 20))
            #y = Variable(mu+torch.randn(batch_size, 20))
            x = np.zeros((100, 2))
            y = np.zeros((100, 2))
            x[:, 1] = np.random.uniform(0, 1, 100)
            y[:, 1] = np.random.uniform(0, 1, 100)

            y[:, 0] = mu
            x = Variable(torch.from_numpy(x).float())
            y = Variable(torch.from_numpy(y).float())
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            fx = model(x)
            fy = model(y)
            #print(x.data.numpy().reshape(-1,28*28).shape)


            loss = loss_fn(fx, fy)
            loss.backward()
            optimizer.step()

            for p in model.parameters():
                p.data.clamp_(-.5, .5)

            print('Loss: ', loss.data[0])
            #print('Loss change: ', (loss.data[0]- prev_loss)/loss.data[0])
            prev_loss = loss.data[0]


    return -loss.data[0]

if __name__ == '__main__':
    loss = []
    m_list = ((np.array(list(range(1, 101))) / 50.0 - 1)).tolist()
    for m in m_list:
         a = train(mu=m)
         print('Mu: ', m)
         print('Dist: ', a)
         loss.append(a)

    df = pd.DataFrame({'Mu': m_list, 'Loss': loss})
    df.to_csv('nn_est2.csv')


