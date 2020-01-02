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
            return ip1


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
            return ip1