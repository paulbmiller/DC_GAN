import os
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt


def preprocess(database, mb_size):
    """
    Preprocessing phase which will return the training and testing dataloaders
    and the number of channels of the images of the chosen dataset.

    Arguments:
        `database`: database ('MNIST', 'CIFAR10', 'CIFAR100')
        `mb_size`: minibatch size
    """
    transform = transforms.ToTensor()

    if database == 'MNIST':
        train_data = torchvision.datasets.MNIST('./data/', download=True,
                                                transform=transform,
                                                train=True)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=mb_size)
        test_data = torchvision.datasets.MNIST('./data/', download=True,
                                               transform=transform,
                                               train=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=mb_size)
        nc = 1

    elif database == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10('./data/', download=True,
                                                  transform=transform,
                                                  train=True)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=mb_size)
        test_data = torchvision.datasets.CIFAR10('./data/', download=True,
                                                 transform=transform,
                                                 train=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=mb_size)
        nc = 3

    elif database == 'CIFAR100':
        train_data = torchvision.datasets.CIFAR100('./data/',
                                                   download=True,
                                                   transform=transform,
                                                   train=True)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=mb_size)
        test_data = torchvision.datasets.CIFAR100('./data/',
                                                  download=True,
                                                  transform=transform,
                                                  train=False)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=mb_size)
        nc = 3

    return train_data, train_loader, test_data, test_loader, nc


def imshow(imgs, epoch, epochs, db, filepath, filename='', save=False):
    """
    Makes a PNG image of the given tensor of MNIST generated images given as
    input and saves it in the results/nb_epochs/ directory.
    """
    imgs = torchvision.utils.make_grid(imgs)
    npimgs = imgs.numpy()
    if db == 'MNIST':
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(npimgs, (1, 2, 0)), cmap='Greys_r')
    elif db == 'CIFAR10' or db == 'CIFAR100':
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    if save:
        plt.savefig(filepath + filename + '_' + str(epoch) + '.png')
    plt.show()


def output_graph(path, filename, save=False):
    """
    This function will get the output text file of a run and in order to
    visualize training data with matplotlib and print the testing results.
    """

    epochs = []
    g_loss = []
    d_loss = []
    d_fake_mean = []
    d_real_acc = []
    d_fake_acc = []

    with open(path + filename, 'r') as f:
        line = f.readline()
        cnt = 1
        while line:
            str_vars = line.split(',')[1:]
            for i in range(len(str_vars)):
                str_vars[i] = str_vars[i].split(' ')[2]
            epochs.append(cnt)
            g_loss.append(float(str_vars[0]))
            d_loss.append(float(str_vars[1]))
            d_fake_mean.append(float(str_vars[2]))
            d_real_acc.append(float(str_vars[3][:-1])/100)
            d_fake_acc.append(float(str_vars[4][:-2])/100)
            cnt += 1
            line = f.readline()

    plt.plot(epochs, g_loss, label='Generator')
    plt.plot(epochs, d_loss, label='Discriminator')
    plt.title('Generator and discriminator losses')
    plt.legend()
    plt.xlabel('Epoch (last epoch is testing)')
    plt.ylabel('Loss')
    if save:
        plt.savefig(path+'losses.png', dpi=200)
    plt.show()

    plt.plot(epochs, d_fake_mean,
             label='Mean output on fake images')
    plt.plot(epochs, d_real_acc,
             label='Accuracy spotting real images')
    plt.plot(epochs, d_fake_acc,
             label='Accuracy spotting fake images')
    plt.plot([0, epochs[-1]], [0.5, 0.5])
    plt.title('Discriminator values')
    plt.legend()
    plt.xlabel('Epoch (last epoch is testing)')
    if save:
        plt.savefig(path+'discriminator_info.png', dpi=200)
    plt.show()


def write_to_file(filepath, str_out):
    mode = 'a' if os.path.exists(filepath) else 'w'
    with open(filepath, mode) as f:
        f.write(str_out+'\n')


class Gen(nn.Module):
    def __init__(self, Z_dim, out_channels, db):
        super(Gen, self).__init__()
        self.db = db
        self.model = nn.Sequential(
                # Input shape: batch_size, Z_dim, 1, 1
                nn.ConvTranspose2d(Z_dim, out_channels*64, kernel_size=5,
                                   bias=False),
                nn.BatchNorm2d(out_channels*64),
                nn.ReLU(True),
                # Shape: batch_size, 1, 5, 5
                nn.ConvTranspose2d(out_channels*64, out_channels*16,
                                   kernel_size=2, stride=2,
                                   bias=False),
                nn.BatchNorm2d(out_channels*16),
                nn.ReLU(True),
                # Shape: batch_size, 1, 10, 10
                nn.ConvTranspose2d(out_channels*16, out_channels*8,
                                   kernel_size=2, bias=False),
                nn.BatchNorm2d(out_channels*8),
                nn.ReLU(True),
                # Shape: batch_size, 16, 11, 11
                nn.ConvTranspose2d(out_channels*8, out_channels*4,
                                   kernel_size=4, stride=2, bias=False),
                nn.BatchNorm2d(out_channels*4),
                nn.ReLU(True),
                # Shape: batch_size, 8, 24, 24
                nn.ConvTranspose2d(out_channels*4, out_channels, kernel_size=5,
                                   bias=False)
                # Shape: batch_size, 1, 28, 28
        )
        if self.db == 'CIFAR10' or self.db == 'CIFAR100':
            self.last_layer = nn.Sequential(
                    nn.ConvTranspose2d(out_channels, out_channels,
                                       kernel_size=5, bias=False),
                    nn.Sigmoid())
        elif self.db == 'MNIST':
            self.last_layer = nn.Sigmoid()

    def forward(self, x):
        return self.last_layer(self.model(x))


class Dis(nn.Module):
    def __init__(self, in_channels, db):
        super(Dis, self).__init__()
        self.db = db
        if self.db == 'CIFAR10' or self.db == 'CIFAR100':
            # shape: batch_size, 3, 32, 32
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=5,
                          bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self.seq = nn.Sequential(
            # shape: batch_size, in, 28, 28
            nn.Conv2d(in_channels, in_channels*4, kernel_size=5, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, in*2, 24, 24
            nn.Conv2d(in_channels*4, in_channels*8, kernel_size=4, stride=2,
                      bias=False),
            nn.BatchNorm2d(in_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, in*4, 11, 11
            nn.Conv2d(in_channels*8, in_channels*16, kernel_size=2,
                      bias=False),
            nn.BatchNorm2d(in_channels*16),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, in*8, 10, 10
            nn.Conv2d(in_channels*16, in_channels*64, kernel_size=2, stride=2,
                      bias=False),
            nn.BatchNorm2d(in_channels*64),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: batch_size, in*16, 5, 5
            nn.Conv2d(in_channels*64, 1, kernel_size=5, bias=False),
            # Shape: batch_size, 1, 1, 1
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.db == 'CIFAR10' or self.db == 'CIFAR100':
            x = self.conv1(x)
        return self.seq(x)


def train(D, G, epochs, device, g_opt, d_opt, Z_dim, filepath, txt_file,
          train_data, train_loader, ep_out, db):
    """
    Training routine which trains the discriminator on every batch and then the
    generator.
    """
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
                                   D_fake_mean, D_right_pred,
                                   D_fake_right_pred)
        print(final_out)
        mode = 'a' if os.path.exists(txt_file) else 'w'
        with open(txt_file, mode) as f:
            f.write(final_out+'\n')

        samples = G(z).detach()
        if db == 'MNIST':
            samples = samples.view(samples.size(0), 1, 28, 28).cpu()
        elif db == 'CIFAR10' or db == 'CIFAR100':
            samples = samples.view(samples.size(0), 3, 32, 32).cpu()
        if epoch % ep_out == 0:
            imshow(samples, epoch, epochs, db, filepath,
                   filename=str(D_fake_mean)[2:11], save=True)
        else:
            imshow(samples, epoch, epochs, db, filepath,
                   filename=str(D_fake_mean)[2:11])


def evaluate(D, G, device, Z_dim, txt_file, test_data, test_loader, db):
    """
    Testing routine which goes through the testing set and the same number
    of generated images. It will save the last batch as samples in results and
    create a string output of the evaluation routine in the results/output.txt
    file.
    """
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
            G_loss = F.binary_cross_entropy(D_fake, one_labels)

            G_loss_run += G_loss.item()
            D_loss_run += D_loss.item()
            D_fake_sum += torch.sum(D_fake).item()

            D_right_pred += torch.sum(D_real > 0.5)
            D_fake_right_pred += torch.sum(D_fake < 0.5)

        D_right_pred = 100*D_right_pred/float(len(test_data))
        D_fake_right_pred = 100*D_fake_right_pred/float(len(test_data))

        str_list = ['Test epoch', 'G_loss: {:0.5f}', 'D_loss: {:0.5f}',
                    'D_fake_mean: {:0.5f}', 'D_real_acc: {:0.3f}%',
                    'D_fake_acc: {:0.3f}%']
        str_out = ', '.join(str_list)
        D_fake_mean = D_fake_sum/len(test_data)
        final_out = str_out.format(G_loss_run/(i+1), D_loss_run/(i+1),
                                   D_fake_mean, D_right_pred,
                                   D_fake_right_pred)
        print(final_out)
        write_to_file(txt_file, final_out)

        samples = G(z).detach()
        if db == 'MNIST':
            samples = samples.view(samples.size(0), 1, 28, 28).cpu()
        elif db == 'CIFAR10' or db == 'CIFAR100':
            samples = samples.view(samples.size(0), 3, 32, 32).cpu()


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        return


def run(db='MNIST', mb_size=64, g_lr=2e-4, d_lr=2e-4, epochs=200, Z_dim=300,
        ep_out=10, CUDA=True):
    """Main function for a typical run of preprocessing/training/testing.

    Arguments are:
        `db`: database (default 'MNIST')
        `mb_size`: minibatch size (default 64)
        `g_lr`: learning rate of the generator (default 2e-4)
        `d_lr`: learning rate of the discriminator (default 2e-4)
        `epochs`: number of epochs of training (default 50)
        `Z_dim`: size of the input noise vector for the generator (default 100)
        `ep_out`: interval of training epochs to save a sample of the last
        batch
        `CUDA`: where we want to use CUDA or not if available (default True)
    """

    train_data, train_loader, test_data, test_loader, nc = preprocess(db,
                                                                      mb_size)
    device = torch.device("cuda" if torch.cuda.is_available() and
                          CUDA else "cpu")
    G = Gen(Z_dim, nc, db).to(device)
    D = Dis(nc, db).to(device)
    g_opt = opt.Adam(G.parameters(), lr=g_lr)
    d_opt = opt.Adam(D.parameters(), lr=d_lr)

    path = 'results/'
    create_dir(path)
    path += db
    path += '/'
    create_dir(path)
    path += str(epochs) + '_epochs/'
    create_dir(path)

    params_str = 'batch_size:{}, g_lr:{}, d_lr:{}, Z_dim:{}\n'.format(
            mb_size, g_lr, d_lr, Z_dim)
    write_to_file(path+'hyperparams.txt', params_str)

    txt_file = path + 'output.txt'
    train(D, G, epochs, device, g_opt, d_opt, Z_dim, path, txt_file,
          train_data, train_loader, ep_out, db)
    evaluate(D, G, device, Z_dim, txt_file, test_data, test_loader, db)

    output_graph(path, 'output.txt', True)


if __name__ == '__main__':
    run('CIFAR10')
