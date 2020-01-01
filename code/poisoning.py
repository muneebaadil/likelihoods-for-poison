import torch
import numpy as np
import os
import pdb
from torchvision.utils import save_image

def compute_loss(model, curr_poison, base_img, target_img, beta_zero):
    with torch.no_grad():
        a = torch.norm(model(curr_poison)[0] - \
            model(target_img)[0])
        b = torch.norm(curr_poison - base_img)
        out = a + beta_zero * b
    return out

def generate_poison(target_img, base_img, model, logger, beta=0.25, max_iters=1000,
                 loss_thres=2.9, lr=500.*255, decay_coeff=.5, min_val=-1.,
                 max_val=1.):
    """
    Generates poison according to Poison Frogs paper.
    https://arxiv.org/abs/1804.00792

    Args:
        target_img: PyTorch tensor of shape (1, 3, H, W)
        base_img: PyTorch tensor of shape (1, 3, H, W)
        model: PyTorch nn.Module of a network
        beta (float), max_iters (int), lr (float): hyper-params of the method.

    Returns:
        poison: PyTorch tensor of shape (1, 3, H, W) containing the poisoned image.
        loss: Loss value of the poisoned image 
    """
    beta_zero = beta * (2048.0 / base_img.numel()) ** 2
    Lp_func = lambda x: torch.norm(model(x)[1] - model(target_img)[1])

    poison = base_img.clone()
    loss = compute_loss(model, poison, base_img, target_img, beta_zero)

    logger.info("Initial loss = {}".format(loss))
    
    for _ in tqdm(range(max_iters)):
        # calculate gradient of Lp w.r.t. x
        poison.requires_grad = True
        norm_val = Lp_func(poison)

        model.zero_grad()
        norm_val.backward()
        grad_Lp = poison.grad.data

        with torch.no_grad():
            # forward step
            poison_hat = poison - lr * grad_Lp
            # backward step
            new_poison = (poison_hat + lr * beta_zero * base_img) / (1 + beta_zero * lr)
            new_poison = torch.clamp(new_poison, min_val, max_val)

            new_loss = compute_loss(model, new_poison, base_img, target_img, beta_zero)
        
            if new_loss < loss_thres: # loss low enough and don't need to optimize further
                # update stuff as final and break out of this optimization.
                poison = new_poison
                loss = new_loss
                logger.info("Optimization done: Loss = {}".format(loss))
                break
            
            if new_loss > loss: # loss is too big than before; don't update stuff.
                lr *= decay_coeff
            else: # loss is lower than before and life is good; update stuff.
                poison, loss = new_poison, new_loss
    logger.info("Final Loss = {}".format(loss))
    return poison, loss


def get_opts():
    import argparse
    from time import gmtime, strftime
    import os, subprocess, sys

    p = argparse.ArgumentParser("Poisoning script for experimentations")
    
    p.add_argument('--debug', action='store_true')
    # data
    p.add_argument('--dataset', action='store', type=str, default='mnist',
                        help='dataset name')
    p.add_argument('--data_path', action='store', type=str,
                        default='../datasets/', help='root path to dataset')
    p.add_argument('--transforms', action='store', type=str, default=None)
    p.add_argument('--n_workers', action='store', type=int, default=4,
                        help='number of workers for data loading')
    # network
    # p.add_argument('--model', action='store', type=str, default='lenet5',
    #                     help='model name')
    # p.add_argument('--print_model', action='store_true',
    #                     help='show model heirarchy on console')
    # p.add_argument('--n_feats', action='store', type=int, default=2,
    #                     help='number of features in the second last layer of the'
    #                     'model')
    # p.add_argument('--n_classes', action='store', type=int,
    #                     default=10, help='number of classes in the dataset')
    p.add_argument('--ckpt_path', action='store', type=str, default=None,
                        help='path to weights file for resuming training process')
    
    # poisoning algorithm
    p.add_argument('--max_targets', type=int, default=-1, help='number of max'
                   ' target images to craft a poison for.')
    p.add_argument('--n_poisons', type=int, default=1, help='number of poisons'
                   'for each target image')

    # logging
    p.add_argument('--log_dir', action='store', type=str,
                        default='../experiments/',
                        help='root path to log experiments in')
    p.add_argument('--exp_name', action='store', type=str,
                        default=strftime("%Y-%m-%d_%H-%M-%S", gmtime()),
                        help='name by which to save experiment')
    # misc
    p.add_argument('--seed', action='store', type=int, default=None,
                        help='integer seed value for reproducibility')
    
    opts = p.parse_args()
    opts.exp_name = 'debug' if opts.debug else opts.exp_name
    opts.save_dir = os.path.join(opts.log_dir, opts.exp_name)

    opts.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    opts.command = ' '.join(sys.argv)

    # set seed
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # cpu/gpu settings config
    if torch.cuda.is_available():
        opts.use_cuda = True
        opts.device = torch.device("cuda")
    else:
        opts.use_cuda = False
        opts.device = torch.device("cpu")

    # making experiment (sub)directories.
    if os.path.exists(opts.save_dir):
        os.system('rm -rf {}'.format(opts.save_dir))
        print("Removing existing experiment directory by the same name.")

    os.makedirs(opts.save_dir)
    os.makedirs(os.path.join(opts.save_dir, 'poisons'))
    opts.folder_names = ['target-{}'.format(x) for x in range(10)]
    for folder_name in opts.folder_names:
        os.makedirs(os.path.join(opts.save_dir, 'poisons', folder_name))

    return opts

def set_logger(save_dir):
    import logging

    class TqdmStream(object):
        @classmethod
        def write(_, msg):
            pass

    logging.basicConfig(stream=TqdmStream)
    logger = logging.getLogger(name='trainer')
    logger.setLevel(logging.DEBUG)

    # create handlers
    # fh = logging.FileHandler(os.path.join(save_dir, 'log.log'))
    # fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(asctime)s; %(levelname)s]: %(message)s'
    )
    ch.setFormatter(formatter)
    # fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    # logger.addHandler(fh)
    return logger

def get_base_idx(target_label, Y_test):
    """
    Returns an index of randomly selected image and its label as a base image
    """
    # pdb.set_trace()
    limit = torch.sum(Y_test != target_label).item()
    idx = torch.randint(high=limit, size=(1,))
    return idx.item()

if __name__ == '__main__':
    import model.net as net
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader

    from tqdm import tqdm
    import pdb

    opts = get_opts()
    logger = set_logger(opts.save_dir)
    logger.info('Experiment folder at %s' % opts.save_dir)

    model = net.Net().to(opts.device)
    model.load_state_dict(torch.load(opts.ckpt_path,
                                     map_location=opts.device))
    t = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)))
    )
    if opts.dataset.lower() == 'mnist':
        data = datasets.MNIST(
            root='../datasets/', train=False, download=True, transform=t
        )
    elif opts.dataset.lower() == 'cifar10':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    loader_temp = DataLoader(data, batch_size=64, shuffle=False,
                        num_workers=opts.n_workers, pin_memory=False)
    X_test, Y_test = [], []
    logger.info("Loading TEST dataset into the memory")
    for (X_, Y_) in tqdm(loader_temp):
        X_test.append(X_)
        Y_test.append(Y_)
    X_test, Y_test = torch.cat(X_test, dim=0), torch.cat(Y_test, dim=0)
    X_test, Y_test = X_test.to(opts.device), Y_test.to(opts.device)

    del loader_temp

    for sample_num, (xtest, ytest) in tqdm(enumerate(zip(X_test, Y_test))):
        # select current image in test set
        target_img = xtest
        target_img.unsqueeze_(0)
        target_label = ytest

        for poison_num in range(opts.n_poisons):
            # select random base from different class than target
            base_idx = get_base_idx(target_label, Y_test)
            base_img, base_label = X_test[base_idx], Y_test[base_idx]
            base_img.unsqueeze_(0)

            logger.info("Crafting poison")
            poison, _ = generate_poison(target_img, base_img, model,
                                     logger)
            poison = poison.squeeze(0)

            # save crafted poison
            filename = '{}_{}_{}.png'.format(base_label, sample_num,
                                             poison_num)
            filepath = os.path.join(opts.save_dir, 'poisons',
                                    opts.folder_names[target_label],
                                    filename)
            save_image(poison, filepath, normalize=True, range=(-1, 1))
            logger.info("Saved image to {}".format(filepath))

        # finetune the network here. (is it really required?)
        if (opts.max_targets > 0) and (opts.max_targets >= sample_num):
            logger.info("Max target limit reached; stopped processing.")
            break

    # X, Y = [], []
    # print("Loading dataset into the memory")
    # for (X_, Y_) in tqdm(loader):
    #     X.append(X_)
    #     Y.append(Y_)
    # X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
    # target_img = X[Y == 4][0] # selecting the first image with label 4.
    # base_img = X[Y == 1][0] # selecting the first iamge with label 1.

    # del X
    # poison_label = torch.tensor([10], dtype=torch.long)
    # Y = torch.cat((Y, poison_label), 0)

    # print("Crafting poison...")
    # poison, loss = do_poisoning(target_img.unsqueeze(0), 
    #              base_img.unsqueeze(0), model)
    
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(ncols=3)

    # poison_ = poison.squeeze(0).squeeze(0)
    # target_img_ = target_img.squeeze(0)
    # base_img_ = base_img.squeeze(0)

    # ax[0].imshow(poison_.detach().numpy(), cmap='gray')
    # ax[1].imshow(target_img_.numpy(), cmap='gray')
    # ax[2].imshow(base_img_.numpy(), cmap='gray')
    # # plt.show()
    # plt.savefig('poisoning.png')

    # print("Plotting Distribution")
    # feats = []
    # for (X, _) in tqdm(loader):
    #     _, feat = model(X)
    #     feats.append(feat.data)
    # _, poison_feat = model(poison)
    # feats.append(poison_feat.data)

    # feats = torch.cat(feats, dim=0).data
    # fig, ax = plt.subplots()
    # for K in range(11):
    #     ax.scatter(feats[Y == K, 0], feats[Y == K, 1],
    #                 label='Class = {}'.format(K))

    # ax.legend()
    # # plt.show()
    # plt.savefig('poisoning_dist.png') 