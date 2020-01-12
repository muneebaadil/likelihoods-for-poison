# used for computing features' statistics

import torch
import numpy as np
import model.net as net
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== 
DEVICE = "cpu"
CKPT_PATH = "../experiments/softmax_mnist/models/epoch-best.model"
DATA_ROOT_PATH = '../datasets/'
BATCH_SIZE = 1
METHOD = 'softmax' # "softmax" OR "lgm"
# ===== 

use_lgm = True if METHOD == 'lgm' else False
device = torch.device(DEVICE)
model = net.MNISTNet(use_lgm=use_lgm).to(device).eval()
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
t = transforms.Compose((
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)))
)
data = datasets.MNIST(
    root=DATA_ROOT_PATH, train=False, download=True, transform=t
)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                    pin_memory=False)

max_feat_val, min_feat_val = None, None
for (x,y) in tqdm(loader):
    x = x.to(device)
    y = y.to(device)
    
    _, feats = model(x)
    _max, _min = feats.max(), feats.min()

    if (max_feat_val is None) or (_max > max_feat_val):
        max_feat_val = _max.item()
    
    if (min_feat_val is None) or (_min < min_feat_val):
        min_feat_val = _min.item()


print("Max: {}; Min: {}".format(max_feat_val, min_feat_val))