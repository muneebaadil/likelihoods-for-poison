import torch
import torch.nn as nn
from torch.autograd import Variable


class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)

        covs = torch.exp(log_covs)  # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1)  # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1)  # eq.(18)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1)  # 1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5 * (tslog_covs + margin_dist)  # eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        # calc of L_lkd
        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5 * torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0 / batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood


class LGMLoss_v0(nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss_v0, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]

        # calc of d_k
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=-1)              # eq.(18)

        # calc of 1 + I(k = z_i)*alpha
        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        #y_onehot = Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0

        margin_dist = torch.mul(dist, y_onehot)
        margin_logits = -0.5 * margin_dist          # eq.(17)
        logits = -0.5 * dist

        # calc of L_lkd
        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        likelihood = (1.0 / batch_size) * cdiff.pow(2).sum(1).sum(0) / 2.0
        return logits, margin_logits, likelihood

class LGMUtils:

    @staticmethod
    def is_anomalous(model, claimed_class, X):
        # we check if the input X which is claiming to be in `claimed_class` is an anomaly
        # in the feature space or not (under Gaussian feature distribution)
        # The assumption is that LGM should return lower likelihood of X  belonging to `claimed_class`
        # if X is poisoned.

        feats = model(X)
        logits, _, _ = model.lgm(feat=feats, label=claimed_class)
        _, predicted = torch.max(logits.data, 1)
        return predicted != claimed_class

    @staticmethod
    def get_likelihood(model, claimed_class, X):

        # we check if the input X which is claiming to be in `claimed_class` is an anomaly
        # in the feature space or not (under Gaussian feature distribution)
        # The assumption is that LGM should return lower likelihood of X  belonging to `claimed_class`
        # if X is poisoned.

        with torch.no_grad():

            # computer 2D features under learned likelihood
            feats = model(X)
            # feature mean of class X is claiming to belong to
            fmean = model.lgm.centers[claimed_class]
            # likelihood (as explained in 1st para of Adversarial Verification section in 4.3)
            # feat and fmean should be size [1,2] tensors
            lkd = torch.exp(-0.5*(feats - fmean).norm(p=2, dim=1)**2)

            return lkd


if __name__ == "__main__":
    # load model and test
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch
    from data.poisons import Poison

    bsize = 4
    tfsm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST('../datasets/', download=True, train=True, transform=tfsm)
    train_loader = DataLoader(trainset, batch_size=bsize, shuffle=False, num_workers=4)
    poisoned_dataset = Poison('../experiments/debug/poisons', tfsm)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=bsize, shuffle=False, num_workers=4)

    # load a model
    from .net import MNISTNet
    import pdb
    import matplotlib.pyplot as plt
    import numpy as np

    model = MNISTNet(use_lgm=True).cuda()
    model.load_state_dict(torch.load('../checkpoints/LGM/LGM40.epoch-60-.model'),
                          strict=False)

    lkd_hist = []
    for X, Y in train_loader:
        X = X.cuda()
        Y = Y.cuda()
        lkd = LGMUtils.get_likelihood(model, Y, X)
        lkd_hist.extend(lkd.cpu().numpy())
        if i*bsize >= 100: break

    plt.ion()
    plt.clf()
    n, b, p = plt.hist(lkd_hist, bins=np.arange(0, 1.05, 0.05), align='mid', facecolor='green', alpha=0.7)
    plt.gca().set_title("Lkd on normal")
    plt.savefig("./hist_normal.jpg")

    plkd_hist = []
    for X, Y, _ in poisoned_loader:
        X = X.cuda()
        Y = Y.cuda()
        lkd = LGMUtils.get_likelihood(model, Y, X)
        plkd_hist.extend(lkd.cpu().numpy())

    plt.ion()
    plt.clf()
    n, b, p = plt.hist(plkd_hist, bins=np.arange(0, 1.05, 0.05), align='mid', facecolor='orange', alpha=0.7)
    plt.gca().set_title("Lkd on poisoned")
    plt.savefig("./hist_poisoned.jpg")

    print("done")
