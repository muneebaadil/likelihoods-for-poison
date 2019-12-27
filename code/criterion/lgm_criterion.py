import torch.nn as nn
from model.lgm import LGMLoss_v0

class LGMCriterion:

    def __init__(self, num_classes, feat_dim, alpha, loss_weight=1.0):

        self.nllloss  = nn.CrossEntropyLoss()
        self.lgm_loss = LGMLoss_v0(num_classes, feat_dim, alpha)
        self.loss_weight = loss_weight

    def __call__(self, features, targets):

        logits, mlogits, likelihood = self.lgm_loss(features, targets)
        loss = self.nllloss(mlogits, targets) + self.loss_weight * likelihood
        return logits, loss
