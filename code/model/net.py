import torch.nn as nn
import torch.nn.functional as F
from .lgm import LGMLoss_v0, LGMLoss


class Net(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, alpha=1.0, use_lgm=False):

        super(Net, self).__init__()
        self.use_lgm = use_lgm

        self.base.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.base.prelu1_1 = nn.PReLU()
        self.base.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.base.prelu1_2 = nn.PReLU()
        self.base.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.base.prelu2_1 = nn.PReLU()
        self.base.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.base.prelu2_2 = nn.PReLU()
        self.base.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.base.prelu3_1 = nn.PReLU()
        self.base.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.base.prelu3_2 = nn.PReLU()
        self.base.preluip1 = nn.PReLU()
        self.base.ip1 = nn.Linear(128 * 3 * 3, 2)

        if not self.use_lgm:
            print("using ip2")
            self.base.ip2 = nn.Linear(2, 10)

        if self.use_lgm:
            self.lgm.lgm1 = LGMLoss_v0(num_classes, feat_dim, alpha)


    def forward(self, x):

        x = self.base.prelu1_1(self.base.conv1_1(x))
        x = self.base.prelu1_2(self.base.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.base.prelu2_1(self.base.conv2_1(x))
        x = self.base.prelu2_2(self.base.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.base.prelu3_1(self.base.conv3_1(x))
        x = self.base.prelu3_2(self.base.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        ip1 = self.base.preluip1(self.base.ip1(x))
        ip2 = self.base.ip2(ip1) if not self.use_lgm else None

        return ip2, ip1
        # return None, ip1
