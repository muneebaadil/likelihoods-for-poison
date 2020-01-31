import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import os
from model.net import MNISTNet
from model.net import CIFARNet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_dataset(dataset_name, data_dir='../datasets/', batch_size=128):

    trainset, train_loader, testset, test_loader = None, None, None, None

    if dataset_name == "mnist":

        tfsm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.MNIST(data_dir, download=True, train=True, transform=tfsm)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testset = datasets.MNIST(data_dir, download=True, train=False, transform=tfsm)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    if dataset_name == "cifar":

        tfsm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.CIFAR10(data_dir, download=True, train=True, transform=tfsm)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testset = datasets.CIFAR10(data_dir, download=True, train=False, transform=tfsm)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainset, train_loader, testset, test_loader


def visualize(feat, labels, epoch, prefix='LGM'):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.xlim(xmin=-7,xmax=7)
    plt.ylim(ymin=-7,ymax=7)
    plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    plt.savefig('./images/%s_loss_epoch=%d.jpg' % (prefix,epoch))

    plt.close()


def test(test_loder, model, opts):

    correct = 0
    total = 0

    ip1_loader = []
    idx_loader = []

    for i, (data, target) in enumerate(test_loder):
        if opts.use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        _, feats = model(data)
        logits, mlogits, likelihood = model.lgm(feats, target)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target.data).sum()

        ip1_loader.append(feats)
        idx_loader.append(target)

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), 0, prefix=opts.ckpt_name)

    print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * correct / total))


def train(train_loader, model, criterion, optimizer, epoch, loss_weight, opts):

    ip1_loader = []
    idx_loader = []

    for i, (data, target) in enumerate(train_loader):
        if opts.use_cuda:
            data = data.cuda()
            target = target.cuda()

        _, feats = model(data)
        logits, mlogits, likelihood = model.lgm(feat=feats, label=target)
        loss = criterion(mlogits, target) + loss_weight * likelihood

        _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ip1_loader.append(feats)
        idx_loader.append(target)

        if (i + 1) % 50 == 0:
            print('Epoch [%d], Iter [%d/%d] Loss: %.4f Acc %.4f'
                  % (epoch, i + 1, len(train_loader), loss.item(), accuracy))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch, prefix=opts.ckpt_name)


def main(opts):

    use_cuda = opts.use_cuda
    batch_size = opts.train_batch_size

    trainset, train_loader, testset, test_loader = get_dataset(opts.dataset, data_dir='../datasets/',
                                                               batch_size=opts.train_batch_size)

    model = MNISTNet(use_lgm=True) if opts.dataset == "mnist" else CIFARNet(use_lgm=True)

    if opts.load_ckpt:
        model.load_state_dict(torch.load(opts.load_ckpt), strict=False)

    criterion = nn.CrossEntropyLoss()
    loss_weight = 0.1

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    optimizer = optim.SGD([
        {'params': model.base.parameters(), 'lr': opts.lr, 'momentum': 0.9, 'weight_decay': 0.0005},
        {'params': model.lgm.parameters(),'lr': 0.001}
    ], lr=0.001, momentum=0.9, weight_decay=0.0005)

    for epoch in range(opts.n_epochs):

        train(train_loader, model, criterion, optimizer, epoch + 1, loss_weight, opts)
        test(test_loader, model, opts)

        if opts.save_ckpt:
            ckpt_name = "%s.epoch-%d-.model" % (opts.ckpt_name, epoch + 1)
            torch.save(model.state_dict(), os.path.join(opts.ckpt_path, ckpt_name))


def get_opts():

    parser = argparse.ArgumentParser(description='Training Script')

    parser.add_argument('--debug', action='store_true', help='flag for '
                        'testing/debugging purposes')

    parser.add_argument('--dataset', action='store', type=str, default='mnist',
                        help='dataset name')
    parser.add_argument('--data_path', action='store', type=str,
                        default='../datasets/', help='root path to dataset')

    # network
    parser.add_argument('--model', action='store', type=str, default='lenet5',
                        help='model name')

    parser.add_argument('--n_epochs', action='store', type=int, default=1,
                        help='training epochs')
    parser.add_argument('--optimizer', action='store', type=str,
                        default='adam', help='optimizing algo for weights update')
    parser.add_argument('--lr', action='store', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--train_batch_size', action='store', type=int,
                        default=128)

    parser.add_argument('--criterion', action='store', type=str,
                        default='CrossEntropyLoss', help='objective function')

    # checkpointing
    parser.add_argument('--ckpt_name', action='store', type=str, help='Name for the model', default='LGM')
    parser.add_argument('--save_ckpt', action='store_true', help='save checkpoint evey epoch')
    parser.add_argument('--ckpt_path', action='store', type=str, default=None,
                        help='path to weights file for resuming training process')
    parser.add_argument('--load_ckpt', action='store', type=str, help='Path to load model as starting point')

    # cpu/gpu config
    parser.add_argument('--cpu', action='store_true', help='run on CPU mode')

    opts = parser.parse_args()
    # cpu/gpu settings config
    if torch.cuda.is_available() and not opts.cpu:
        opts.use_cuda = True
        opts.device = torch.device("cuda")
    else:
        opts.use_cuda = False
        opts.device = torch.device("cpu")

    return opts


if __name__ == '__main__':

    opts = get_opts()
    main(opts)
