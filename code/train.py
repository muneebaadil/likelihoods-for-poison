import data as dt
import model as m
import criterion as cr
import logger as lg

from tqdm import tqdm


def get_opts():
    import argparse
    from time import gmtime, strftime
    import torch
    import os
    import subprocess
    import yaml
    import sys

    parser = argparse.ArgumentParser(description='Training Script')

    # config file (optional)
    # REMOVING CONFIG PATH OPTION FOR NOW
    # parser.add_argument('--config_path', action='store', type=str, default='.')
    parser.add_argument('--debug', action='store_true', help='flag for '
                        'testing/debugging purposes')
    # data
    parser.add_argument('--dataset', action='store', type=str, default='mnist',
                        help='dataset name')
    parser.add_argument('--data_path', action='store', type=str,
                        default='../datasets/', help='root path to dataset')
    parser.add_argument('--transforms', action='store', type=str, default=None)
    parser.add_argument('--n_workers', action='store', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='do not shuffle during training')
    # network
    parser.add_argument('--model', action='store', type=str, default='lenet5',
                        help='model name')
    parser.add_argument('--print_model', action='store_true',
                        help='show model heirarchy on console')
    parser.add_argument('--n_classes', action='store', type=int,
                        default=10, help='number of classes in the dataset')
    parser.add_argument('--ckpt_path', action='store', type=str, default=None,
                        help='path to weights file for resuming training process')
    # training
    parser.add_argument('--n_epochs', action='store', type=int, default=1,
                        help='training epochs')
    parser.add_argument('--optimizer', action='store', type=str,
                        default='adam', help='optimizing algo for weights update')
    parser.add_argument('--lr', action='store', type=float, default=.02,
                        help='learning rate')
    parser.add_argument('--lr_scheduler', action='store', type=str,
                        default='none', help='learning rate scheduler algo')
    parser.add_argument('--step_size', action='store', type=int, default=1,
                        help='step size for stepLR')
    parser.add_argument('--gamma', action='store', type=float, default=0.1,
                        help='gamma param for stepLR algo')
    parser.add_argument('--train_batch_size', action='store', type=int,
                        default=32)
    parser.add_argument('--val_batch_size', action='store', type=int,
                        default=32)
    parser.add_argument('--criterion', action='store', type=str,
                        default='CrossEntropyLoss', help='objective function')
    # logging
    parser.add_argument('--log_dir', action='store', type=str,
                        default='../experiments/',
                        help='root path to log experiments in')
    parser.add_argument('--hist_freq', action='store', type=int, default=-1,
                        help='tensorboard histogram logging frequency')
    parser.add_argument('--exp_name', action='store', type=str,
                        default=strftime("%Y-%m-%d_%H-%M-%S", gmtime()),
                        help='name by which to save experiment')
    parser.add_argument('--save_every_ckpt', action='store_true',
                        help='save model checkpoint every epoch')
    parser.add_argument('--log_every', action='store', type=int, default=1,
                        help='iterations step size of logging')
    # cpu/gpu config
    parser.add_argument('--cpu', action='store_true', help='run on CPU mode')
    parser.add_argument('--gpu_ids', action='store', type=str, default='0',
                        help='comma seperated GPU IDs to run training on')
    # misc
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='integer seed value for reproducibility')
    parser.add_argument('--validate_only', action='store_true',
                        help='only validate the model on valset')

    opts = parser.parse_args()

    # cpu/gpu settings config
    if torch.cuda.is_available() and not opts.cpu:
        opts.use_cuda = True
        opts.device = torch.device("cuda")
    else:
        opts.use_cuda = False
        opts.device = torch.device("cpu")

    opts.gpu_ids = [int(x) for x in opts.gpu_ids.split(',')]
    opts.n_gpus = len(opts.gpu_ids)

    # set seed
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # creation of logging directories
    opts.exp_name = 'debug' if opts.debug else opts.exp_name
    opts.save_dir = os.path.join(opts.log_dir, opts.exp_name)
    opts.save_dir_model = os.path.join(opts.save_dir, 'models')
    opts.save_dir_result = os.path.join(opts.save_dir, 'results')
    opts.save_dir_tensorboard = os.path.join(opts.save_dir, 'tensorboard')
    for d in [opts.save_dir, opts.save_dir_model, opts.save_dir_result,
              opts.save_dir_tensorboard]:
        if os.path.exists(d):
            os.system('rm -rf {}'.format(d))
        os.makedirs(d)

    opts.shuffle = True if not opts.no_shuffle else False
    opts.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    opts.command = ' '.join(sys.argv)

    # overwrite by config file settings if supplied
    # if opts.config_path != '.':
    #     cfg = yaml.load(open(opts.config_path))
    #     for k, v in cfg.items():
    #         setattr(opts, k, v)
    return opts


def get_optimizer(opts, model):
    from torch import optim
    # set optimizer
    if opts.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr,
                              momentum=opts.momentum)
    elif opts.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    else:
        raise NotImplementedError()

    # set lr scheduler
    if opts.lr_scheduler == 'stepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.step_size, gamma=opts.gamma
        )
    elif opts.lr_scheduler == 'none':
        lr_scheduler = None
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(loader, model, criterion, optimizer, lr_scheduler, logger, device,
          curr_epoch, curr_global_iter, n_epochs):

    logger.logger.info("Epoch [{} / {}]".format(curr_epoch + 1, n_epochs))
    loader_iterable = tqdm(loader)
    for (X_train, Y_train) in loader_iterable:
        X_train, Y_train = X_train.to(device), Y_train.to(device)

        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_iter(curr_epoch, curr_global_iter, loss.data,
                        get_lr(optimizer), model=model)
        curr_global_iter += 1
        loader_iterable.set_postfix(
            loss=loss.data.tolist(), lr=get_lr(optimizer))

    logger.log_epoch(curr_epoch, len(loader))

    return curr_global_iter, curr_epoch < n_epochs


def validate(loader, model, criterion, lr_scheduler, logger, device,
             curr_epoch, curr_global_iter):
    model.eval()

    logger.logger.info("Validating...")
    loader_iterable = tqdm(loader)
    for (X_val, Y_val) in loader_iterable:
        X_val, Y_val = X_val.to(device), Y_val.to(device)
        Y_pred = model(X_val)
        loss = criterion(Y_pred, Y_val)

        logger.log_iter(curr_epoch, curr_global_iter, loss.data, None, False)
        loader_iterable.set_postfix(loss=loss.data.tolist())

    logger.log_epoch(curr_epoch, len(train_loader), False, model=model)

    if lr_scheduler:
        lr_scheduler.step()


opts = get_opts()
logger = lg.get_logger(opts)
train_loader, val_loader = dt.get_loaders(opts)
model = m.get_model(opts, logger)
criterion = cr.get_criterion(opts)
optimizer, lr_scheduler = get_optimizer(opts, model)

terminate = False
curr_epoch, curr_global_iter = 0, 0

logger.logger.info('Modules configured; training now...')

while not terminate:
    curr_global_iter, terminate = train(
        train_loader, model, criterion, optimizer, lr_scheduler, logger,
        opts.device, curr_epoch, curr_global_iter, opts.n_epochs
    )
    validate(val_loader, model, criterion, lr_scheduler, logger, opts.device,
             curr_epoch, curr_global_iter)

    curr_epoch += 1
