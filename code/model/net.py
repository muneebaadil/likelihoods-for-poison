import torch
import torch.nn as nn
from .lgm import LGMLoss_v0, LGMLoss


class MNISTNet(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, alpha=1.0, use_lgm=False):

        super(MNISTNet, self).__init__()
        self.use_lgm = use_lgm

        modules = [
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.PReLU(),
            nn.Linear(128 * 3 * 3, 2)
        ]

        if self.use_lgm:
            self.lgm = LGMLoss_v0(num_classes, feat_dim, alpha)
        else:
            self.classifier = nn.Linear(2, 10)

        self.base = nn.Sequential(*modules)

    def forward(self, x):

        # 2d features
        ip1 = self.base(x)

        if not self.use_lgm:
            # 10-d clf output
            ip2 = self.classifier(ip1)
            return ip2, ip1
        else:
            return None, ip1


class CIFARNet(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, alpha=1.0, use_lgm=False):

        super(CIFARNet, self).__init__()
        self.use_lgm = use_lgm

        modules = [
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.PReLU(),
            nn.Linear(128 * 4 * 4, 2)
        ]

        if self.use_lgm:
            self.lgm = LGMLoss_v0(num_classes, feat_dim, alpha)
        else:
            self.classifier = nn.Linear(2, 10)

        self.base = nn.Sequential(*modules)

    def forward(self, x):

        # 2d features
        ip1 = self.base(x)

        if not self.use_lgm:
            # 10-d clf output
            ip2 = self.classifier(ip1)
            return ip2, ip1
        else:
            return None, ip1


class VGG(nn.Module):

    def __init__(self, vgg_name, use_lgm=False):

        super(VGG, self).__init__()

        self.base = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        self.use_lgm = use_lgm

        if self.use_lgm:
            self.lgm = LGMLoss_v0(10, 50, alpha=1.0)
        else:
            self.classifer = nn.Linear(50, 10)

    def forward(self, x):

        ip1 = self.base(x)
        # ip1 = ip1.view(ip1.size(0), -1)
        if not self.use_lgm:
            ip2 = self.classifier(ip1)
            return ip2, ip1
        else:
            return None, ip1

    def _make_layers(self, cfg):

        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(512, 50)]
        return nn.Sequential(*layers)
