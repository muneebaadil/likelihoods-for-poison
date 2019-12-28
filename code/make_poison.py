import data as dt
import model as m
import logger as lg

import torch
import torch.nn as nn

import tqdm as tqdm
import pdb

import argparse
from time import gmtime, strftime
import torch
import os
import subprocess
import yaml
import sys

p = argparse.ArgumentParser(description='Poisoning via PoisonFrogs paper')
# parser.add_argument('--config_path', action='store', type=str, default='.')
p.add_argument('--debug', action='store_true', help='flag for '
                    'testing/debugging purposes')
# data
p.add_argument('--dataset', action='store', type=str, default='mnist',
                    help='dataset name')
p.add_argument('--data_path', action='store', type=str,
                    default='../datasets/', help='root path to dataset')
p.add_argument('--transforms', action='store', type=str, default=None)
p.add_argument('--n_workers', action='store', type=int, default=4,
                    help='number of workers for data loading')

# network
p.add_argument('--model', action='store', type=str, default='lenet5',
                    help='model name')
p.add_argument('--print_model', action='store_true',
                    help='show model heirarchy on console')
p.add_argument('--n_feats', action='store', type=int, default=2,
                    help='number of features in the second last layer of the'
                    'model')
p.add_argument('--n_classes', action='store', type=int,
                    default=10, help='number of classes in the dataset')
p.add_argument('--ckpt_path', action='store', type=str, default=None,
                    help='path to weights file for resuming training process')

# poisoning
p.add_argument('--max_iters', action='store', type=int, default=1000,
                help='maximum iterations for PF algorithm')
p.add_argument('--beta', action='store', type=float, default=0.25,
                help='Beta parameter for PF algorithm')
p.add_argument('--lr_pf', action='store', type=float, default=0.01,
                help='learning rate for PF algorithm')

# logging
p.add_argument('--log_dir', action='store', type=str,
                    default='../experiments/',
                    help='root path to log experiments in')
p.add_argument('--exp_name', action='store', type=str,
                    default=strftime("%Y-%m-%d_%H-%M-%S", gmtime()),
                    help='name by which to save experiment')

# cpu/gpu config
p.add_argument('--cpu', action='store_true', help='run on CPU mode')
p.add_argument('--gpu_ids', action='store', type=str, default='0',
                    help='comma seperated GPU IDs to run training on')
# misc
p.add_argument('--seed', action='store', type=int, default=None,
                    help='integer seed value for reproducibility')
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

opts.exp_name = 'debug' if opts.debug else opts.exp_name
opts.save_dir = os.path.join(opts.log_dir, opts.exp_name)
opts.save_dir_result = os.path.join(opts.save_dir, 'results')

opts.shuffle = True if not opts.no_shuffle else False
opts.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
opts.command = ' '.join(sys.argv)


