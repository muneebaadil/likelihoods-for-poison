from importlib import import_module
import torch.nn as nn

def get_criterion(opts):

    if opts.criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif opts.criterion == 'lgm':
        from .lgm_criterion import LGMCriterion
        criterion = LGMCriterion(10, 2, 1.0, loss_weight=1.0)
    else:
        raise NotImplementedError()

    if opts.use_cuda:
        criterion = criterion.cuda()

    return criterion
