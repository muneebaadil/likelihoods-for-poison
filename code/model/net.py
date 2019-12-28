import torch.nn as nn
import torch.nn.functional as F
from .lgm import LGMLoss_v0, LGMLoss


class Net(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, alpha=1.0, use_lgm=False):

        super(Net, self).__init__()
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
            #pass
            self.lgm = LGMLoss_v0(num_classes, feat_dim, alpha)
        else:
            modules.append(nn.Linear(2, 10))

        self.base = nn.Sequential(*modules)

    def forward(self, x):

        ip1 = self.base(x)
        return ip1
