# used for computing features' statistics

import torch
import numpy as np
import model.net as net
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== 
DEVICE = "cpu"
CKPT_PATH = "../experiments/lgm_mnist/lgm-model"
DATA_ROOT_PATH = '../datasets/'
BATCH_SIZE = 1
METHOD = 'lgm' # "softmax" OR "lgm"
NUM_FEATS = 2
DATASET = 'mnist' # mnist or cifar10
# ===== 

def where(cond, x_1, x_2):
    out = torch.zeros_like(x_1)
    out = x_2
    out[cond] = x_1[cond]
    return out

use_lgm = True if METHOD == 'lgm' else False
device = torch.device(DEVICE)

if DATASET == 'mnist':
    model = net.MNISTNet(use_lgm=use_lgm).to(device).eval()
    t = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)))
    )
    data = datasets.MNIST(
        root=DATA_ROOT_PATH, train=False, download=True, transform=t
    )

elif DATASET == 'cifar10':
    model = net.VGG('vgg16', use_lgm=use_lgm).to(device).eval()

    t = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    )
    data = datasets.CIFAR10(
        root=DATA_ROOT_PATH, train=False, download=True, transform=t
    )

model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                    pin_memory=False)
maxs = torch.tensor([-float("inf")] * NUM_FEATS)
mins = torch.tensor([float("inf")] * NUM_FEATS)

with torch.no_grad():
    max_feat_val, min_feat_val = None, None
    for (x,y) in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        
        _, feats = model(x)
        curr_maxs = feats.max(dim=0)[0]
        curr_mins = feats.min(dim=0)[0]

        maxs = where(maxs > curr_maxs, maxs, curr_maxs)
        mins = where(mins < curr_mins, mins, curr_mins)
        # _max, _min = feats.max(), feats.min()

        # if (max_feat_val is None) or (_max > max_feat_val):
        #     max_feat_val = _max.item()
        
        # if (min_feat_val is None) or (_min < min_feat_val):
        #     min_feat_val = _min.item()

# save max and min vals
torch.save(maxs, 'maxs_{}_{}.pt'.format(DATASET, METHOD))
torch.save(mins, 'mins_{}_{}.pt'.format(DATASET, METHOD))