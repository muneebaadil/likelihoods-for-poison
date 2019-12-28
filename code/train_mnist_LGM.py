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
from model.net import Net

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize(feat, labels, epoch):

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
    plt.savefig('./images/LGM_loss_epoch=%d.jpg' % epoch)

    plt.close()


def test(test_loder, model, use_cuda):

    correct = 0
    total = 0
    for i, (data, target) in enumerate(test_loder):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        feats = model(data)
        logits, mlogits, likelihood = model.lgm(feats, target)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target.data).sum()

    print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * correct / total))


def train(train_loader, model, criterion, optimizer, epoch, loss_weight, use_cuda):

    ip1_loader = []
    idx_loader = []

    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()

        feats = model(data)
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

            if opts.save_ckpt:

                save_name = "%s.epoch-%d-.model" % (epoch, opts.save_name)
                torch.save(model.cpu().state_dict(), os.path.join(
                               opts.ckpt_path, save_name))


    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)


def main(opts):

    use_cuda = opts.use_cuda
    batch_size = opts.train_batch_size

    # Dataset
    trainset = datasets.MNIST('../datasets/', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.MNIST('../datasets/', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Net(use_lgm=True)
    criterion = nn.CrossEntropyLoss()
    loss_weight = 0.1

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    optimizer = optim.SGD([
        {'params': model.base.parameters(), 'lr': opts.lr, 'momentum': 0.9, 'weight_decay': 0.0005},
        {'params': model.lgm.parameters(),'lr': 0.01}
    ], lr=0.001, momentum=0.9, weight_decay=0.0005)

    for epoch in range(opts.n_epochs):

        train(train_loader, model, criterion, optimizer, epoch + 1, loss_weight, use_cuda)
        test(test_loader, model, use_cuda)


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
    parser.add_argument('--save_name', action='store', type=str, help='Name for the model')
    parser.add_argument('--save_ckpt', action='store_true', help='save checkpoint evey epoch')
    parser.add_argument('--ckpt_path', action='store', type=str, default=None,
                        help='path to weights file for resuming training process')

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
