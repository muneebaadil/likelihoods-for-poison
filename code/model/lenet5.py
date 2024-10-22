# ============================================================================
# TAKEN FROM https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py
# ============================================================================

import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self, n_feats, n_classes):
        super(LeNet5, self).__init__()
        self.n_feats = n_feats
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.feats = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, n_feats)),
            ('relu6', nn.ReLU()),
        ]))

        self.out = nn.Linear(n_feats, n_classes)

    def forward(self, img):
        temp = self.convnet(img)
        temp = temp.view(temp.size(0), -1)
        feats = self.feats(temp)
        output = self.out(feats)
        return output, feats
