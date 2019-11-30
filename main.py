import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

mb_size = 64
g_lr = 2e-4
d_lr = 2e-4
epochs = 250
CUDA = True
device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST('./data/', download=True,
                                        transform=transform, train=True)
train_loader = DataLoader(train_data, shuffle=True, batch_size=mb_size)
data_iter = iter(train_loader)

test_data = torchvision.datasets.MNIST('./data/', download=True,
                                       transform=transform, train=False)
test_loader = DataLoader(test_data, shuffle=False, batch_size=mb_size)

imgs, labels = data_iter.next()

txt_file = 'results/' + str(epochs) + '_epochs/' + 'output.txt'


def imshow(imgs, epoch, filename='', save=False,
           save_dir='results/' + str(epochs) + '_epochs/'):
    imgs = torchvision.utils.make_grid(imgs)
    npimgs = imgs.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimgs, (1, 2, 0)), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    if save:
        plt.savefig(save_dir + filename + '_' + str(epoch) + '.png')
    plt.show()


Z_dim = 100
X_dim = imgs.view(imgs.size(0), -1).size(1)
X = int(np.sqrt(X_dim))


class Flatten(torch.nn.Module):
    """
    Flatten a convolution block into a simple vector.
    Replaces the flattening line (view) often found into forward() methods of
    networks. This makes it easier to navigate the network with introspection.
    """
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Unflatten(torch.nn.Module):
    """
    Unflatten a vector during the forward pass in order to run convolutions.
    """
    def forward(self, x):
        x = x.view(x.size(0), 1, 10, 10)
        return x


class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.model = nn.Sequential(
                # Input shape: batch_size, Z_dim, 1, 1
                nn.ConvTranspose2d(Z_dim, 16, kernel_size=5, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                # Shape: batch_size, 1, 5, 5
                nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,
                                   bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                # Shape: batch_size, 1, 10, 10
                nn.ConvTranspose2d(8, 4, kernel_size=2, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(True),
                # Shape: batch_size, 16, 11, 11
                nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                # Shape: batch_size, 8, 24, 24
                nn.ConvTranspose2d(2, 1, kernel_size=5, bias=False),
                nn.Sigmoid()
                # Shape: batch_size, 1, 28, 28
        )

    def forward(self, x):
        return self.model(x)


G = Gen().to(device)


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.model = nn.Sequential(
            # Input shape: batch_size, 1, 28, 28
            nn.Conv2d(1, 2, kernel_size=5, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, 8, 24, 24
            nn.Conv2d(2, 4, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, 16, 11, 11
            nn.Conv2d(4, 8, kernel_size=2, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, 1, 10, 10
            nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, 1, 5, 5
            nn.Conv2d(16, 1, kernel_size=5, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


D = Dis().to(device)

g_opt = opt.Adam(G.parameters(), lr=g_lr)
d_opt = opt.Adam(D.parameters(), lr=d_lr)

for epoch in range(1, epochs+1):
    G_loss_run = 0.0
    D_loss_run = 0.0
    D_fake_sum = 0.0
    D_right_pred = 0
    D_fake_right_pred = 0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        b_size = data.size(0)

        one_labels = torch.ones(b_size, 1, 1, 1).to(device)
        zero_labels = torch.zeros(b_size, 1, 1, 1).to(device)

        z = torch.randn(b_size, Z_dim, 1, 1).to(device)

        D_real = D(data)
        D_fake = D(G(z))

        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)

        D_loss = D_real_loss + D_fake_loss

        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()

        z = torch.randn(b_size, Z_dim, 1, 1).to(device)
        D_fake = D(G(z))
        G_loss = F.binary_cross_entropy(D_fake, one_labels)

        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()

        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()
        D_fake_sum += torch.sum(D_fake).item()

        with torch.no_grad():
            D_right_pred += torch.sum(D_real > 0.5)
            D_fake_right_pred += torch.sum(D_fake < 0.5)

    D_right_pred = 100*D_right_pred/float(len(train_data))
    D_fake_right_pred = 100*D_fake_right_pred/float(len(train_data))

    str_list = ['Epoch: {}', 'G_loss: {:0.5f}', 'D_loss: {:0.5f}',
                'D_fake_mean: {:0.5f}', 'D_real_acc: {:0.3f}%',
                'D_fake_acc: {:0.3f}%']
    str_out = ', '.join(str_list)
    D_fake_mean = D_fake_sum/len(train_data)
    final_out = str_out.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1),
                               D_fake_mean, D_right_pred, D_fake_right_pred)
    print(final_out)
    f = open(txt_file, 'a')
    f.write(final_out+'\n')
    f.close()

    samples = G(z).detach()
    samples = samples.view(samples.size(0), 1, 28, 28).cpu()
    if epoch % 10 == 0:
        imshow(samples, epoch, filename=str(D_fake_mean)[2:11], save=True)
    else:
        imshow(samples, epoch, filename=str(D_fake_mean)[2:11])


with torch.no_grad():
    G_loss_run = 0.0
    D_loss_run = 0.0
    D_fake_sum = 0.0
    D_right_pred = 0
    D_fake_right_pred = 0
    G.eval()
    D.eval()
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        b_size = data.size(0)

        one_labels = torch.ones(b_size, 1, 1, 1).to(device)
        zero_labels = torch.zeros(b_size, 1, 1, 1).to(device)

        z = torch.randn(b_size, Z_dim, 1, 1).to(device)

        D_real = D(data)
        D_fake = D(G(z))

        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)

        D_loss = D_real_loss + D_fake_loss

        z = torch.randn(b_size, Z_dim, 1, 1).to(device)
        D_fake = D(G(z))
        G_loss = F.binary_cross_entropy(D_fake, one_labels)

        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()
        D_fake_sum += torch.sum(D_fake).item()

        D_right_pred += torch.sum(D_real > 0.5)
        D_fake_right_pred += torch.sum(D_fake < 0.5)

    D_right_pred = 100*D_right_pred/float(len(test_data))
    D_fake_right_pred = 100*D_fake_right_pred/float(len(test_data))

    str_list = ['Test epoch: {}', 'G_loss: {:0.5f}', 'D_loss: {:0.5f}',
                'D_fake_mean: {:0.5f}', 'D_real_acc: {:0.3f}%',
                'D_fake_acc: {:0.3f}%']
    str_out = ', '.join(str_list)
    D_fake_mean = D_fake_sum/len(test_data)
    final_out = str_out.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1),
                               D_fake_mean, D_right_pred, D_fake_right_pred)
    print(final_out)
    f = open(txt_file, 'a')
    f.write(final_out+'\n')
    f.close()

    samples = G(z).detach()
    samples = samples.view(samples.size(0), 1, 28, 28).cpu()
