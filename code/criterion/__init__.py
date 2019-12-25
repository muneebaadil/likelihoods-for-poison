from importlib import import_module
import torch.nn as nn

def get_criterion(opts):
    if opts.criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    if opts.use_cuda:
        criterion = criterion.cuda()

    return criterion
