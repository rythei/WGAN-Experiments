import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from emd import emd
import numpy as np
import pandas as pd

'''
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.batch_size = 50

        self.critic = nn.Sequential(
            nn.Linear(28 * 28, 16 * 16),
            nn.ReLU(),
            nn.Linear(16 * 16, 10 * 10),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        fx = self.critic(x)

        return fx
'''

class WAE(nn.Module):
    def __init__(self):
        super(WAE, self).__init__()

        self.image_dim = 28 # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 20
        self.batch_size = 50

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 16*16),
            nn.ReLU(),
            nn.Linear(16*16, 10*10),
            nn.ReLU(),
            nn.Linear(100, self.latent_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.sigma = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 16*16),
            nn.ReLU(),
            nn.Linear(16*16, 28*28)
        )


    def forward(self, x):
        x = x.view(-1, 28*28)
        z = self.encoder(x)
        mu_z = self.mu(z)
        log_sigma_z = self.sigma(z)
        sample_z = mu_z + log_sigma_z.exp()*Variable(torch.randn(self.batch_size, self.latent_dim))
        x_hat = self.decoder(sample_z)

        return mu_z, log_sigma_z, x_hat


def wae_loss(x, x_hat, mu, logvar, batch_size=128):

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KL divergence
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size
    EMDist = emd(x, x_hat)

    return EMDist + KLD


def train(num_epochs = 100, batch_size = 128, learning_rate = 1e-4):
    train_dataset = dsets.MNIST(root='./data/',  #### testing that it works with MNIST data
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    #bvae = AE()
    #bvae.batch_size = batch_size

    wae = WAE()
    wae.batch_size = batch_size

    optimizer = torch.optim.Adam(wae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            x = Variable(images)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            mu_z, log_sigma_z, x_hat = wae(x)

            loss = wae_loss(x, x_hat, mu_z, 2*log_sigma_z, batch_size=batch_size)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

        torch.save(wae.state_dict(), 'wae-test-model.pkl')

if __name__ == '__main__':
    train()
