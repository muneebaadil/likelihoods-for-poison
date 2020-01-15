import torch
import numpy as np
import os
import pdb
from torchvision.utils import save_image
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_opts():
    import argparse
    from time import gmtime, strftime
    import os, subprocess, sys

    p = argparse.ArgumentParser("Poisoning script for experimentations")
    
    p.add_argument('--debug', action='store_true')

    # method and weights file.
    p.add_argument('--method', action='store', type=str, default='softmax',
                   help='[softmax|lgm] loss function the prior net was trained on')
    p.add_argument('--ckpt_path', action='store', type=str, default=None,
                    help='path to weights file for resuming training process')
    p.add_argument('--normalize_feats', action='store_true')
    # data
    p.add_argument('--dataset', action='store', type=str, default='mnist',
                        help='dataset name')
    p.add_argument('--data_path', action='store', type=str,
                        default='../datasets/', help='root path to dataset')
    p.add_argument('--transforms', action='store', type=str, default=None)
    p.add_argument('--n_workers', action='store', type=int, default=4,
                        help='number of workers for data loading')
    
    # poisoning algorithm hyperparams.
    p.add_argument('--poisoning_strength', type=int, default=.1, help='fraction'
                   'of dataset which is allowed to be poisoned. number in the'
                   ' range [0, 1]')
    p.add_argument('--poison_lr', type=float, default=255*500., help='learning '
                   'rate for poisoning algorithm.')
    p.add_argument('--overlay', action='store_true', help='add overlay before'
                   ' optimizing for poison.')
    p.add_argument('--overlay_alpha', action='store', type=float,
                   default=0.2, help='strength'
                   ' for overlaying target image onto base image.')
    p.add_argument('--base_strategy', type=str, default='random',
                   help='[random|closest] strategy for selecting base image for'
                   ' each target image.')
    p.add_argument('--dist_neighbours', type=str, default='softmax',
                    help='[softmax|lgm] which NNs (in feature distribution space)'
                    'to use for selecting closest base.')
    p.add_argument('--max_poisons', type=int, default=-1, help='upper limit'
                   ' on number of poisons to create. (for debugging purposes.)')
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
    opts.use_lgm = True if opts.method.lower() == 'lgm' else False
    opts.exp_name = 'debug' if opts.debug else opts.exp_name
    opts.save_dir = os.path.join(opts.log_dir, opts.exp_name)

    opts.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    opts.command = ' '.join(sys.argv)

    # set seed
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opts.seed)

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

    if opts.base_strategy == 'random':
        opts.nn_dict = {}
    elif opts.base_strategy == 'closest':
        if opts.dist_neighbours == 'softmax':
            # nearest neighbours for each class in softmax 
            # feature distribution. 
            opts.nn_dict = {0: [4, 5], 1: [3, 9], 2: [6, 7], 3: [1, 8],
                        4: [0, 9], 5: [0, 6], 6: [2, 5], 7: [2, 8],
                        8: [3, 7], 9: [1, 4]}
        elif opts.dist_neightbours == 'lgm':
            # nearest neighbours for each class in lgm
            # feature distribution
            opts.nn_dict = {0: [6, 7], 1: [2, 7, 5], 2: [1, 7, 8, 3, 5],
                        3: [5, 2, 4], 4: [3, 8, 9], 5: [1, 2, 3],
                        6: [0, 7, 8, 9], 7: [0, 6, 8, 2, 1],
                        8: [2, 7, 6, 8, 4], 9: [6, 8, 4]}
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    # feature scaling.
    if opts.method == 'softmax':
        opts.feats_max = torch.tensor([334.7420, 209.5980])
        opts.feats_min = torch.tensor([-259.4084, -381.4945])
        opts.feats_max = opts.feats_max.to(opts.device)
        opts.feats_min = opts.feats_min.to(opts.device)

        opts.feats_range = opts.feats_max - opts.feats_min
    elif opts.method == 'lgm':
        opts.feats_max = torch.tensor([6.0089, 5.3333])
        opts.feats_min = torch.tensor([-6.5698, -5.4916])
        opts.feats_max = opts.feats_max.to(opts.device)
        opts.feats_min = opts.feats_min.to(opts.device)
        opts.feats_range = opts.feats_max - opts.feats_min
    else:
        raise NotImplementedError()

    return opts

def model_normalized(model, x, min, max):
    a, b = model(x)
    range = max - min
    
    # renormalize to [-1, +1]
    if opts.normalize_feats:
        b = (((b - min) / range) * 2) - 1 
    return a, b

def compute_loss(model, curr_poison, base_img, target_img, beta_zero):
    with torch.no_grad():
        a = torch.norm(
            model_normalized(model, curr_poison, opts.feats_min, opts.feats_max)[1] - \
            model_normalized(model, target_img, opts.feats_min, opts.feats_max)[1]
        )
        b = torch.norm(curr_poison - base_img)
        out = a + beta_zero * b
    return out

def generate_poison(target_img, base_img, model, logger, beta=0.25, max_iters=1000,
                 loss_thres=1e-4, lr=500.*255, decay_coeff=.5, min_val=-1.,
                 max_val=1., overlay=False, overlay_alpha=0.2):
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

    def Lp_func(x):
        alpha = model_normalized(model, x, opts.feats_min, opts.feats_max)[1]
        beta = model_normalized(model, target_img, opts.feats_min, opts.feats_max)[1]
        return torch.norm(alpha - beta)

    poison = base_img.clone()
    if overlay:
        with torch.no_grad():
            poison = poison + overlay_alpha * target_img
    loss = compute_loss(model, poison, base_img, target_img, beta_zero)

    logger.info("Initial loss = {}".format(loss))
    
    for _ in trange(max_iters, leave=False):
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
            
            if new_loss >= loss: # loss is too big than before; don't update stuff.
                lr *= decay_coeff
                # lr *= .1
            else: # loss is lower than before and life is good; update stuff.
                # print("Loss updated")
                poison, loss = new_poison, new_loss
    logger.info("Final Loss = {}".format(loss))
    return poison, loss

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

def get_base_class_random(target_label, Y_test):
    """
    Returns a randomly selected base class
    """
    candidate_labels = Y_test[Y_test != target_label]
    base_class = np.random.choice(candidate_labels.cpu().numpy())
    return base_class

def get_base_class_closest(target_label, Y_test, nn_dict):
    """
    Returns a NN based base class.
    """
    candidate_classes = nn_dict[target_label.item()]
    base_class = np.random.choice(candidate_classes.cpu().numpy())
    return base_class

def get_random_instance(label, X_test, Y_test):
    """
    Given a label, returns random image of that label
    """
    idx = np.random.randint(
        low=0, high=np.sum((Y_test == label).cpu().numpy())
    )
    img = X_test[Y_test == label][idx]
    return img, idx

def get_features(X, model, logger):
    out = []
    logger.info("Generating features")
    with torch.no_grad():
        for example in tqdm(X):
            _, feat = model_normalized(model, example.unsqueeze(0),
                                       opts.feats_min, opts.feats_max)
            out.append(feat)
    return torch.cat(out)

def draw_features(clean_features, clean_labels, poisoned_features, 
                  poisoned_bases, poisoned_targets, save_dir, logger,
                  n_classes=10):
    if (clean_features.shape[1] != 2):
        raise NotImplementedError("Draw feautres only implemented with"
                                    "2 features")
    fig, ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    for K in range(n_classes):
        ax.scatter(
            clean_features[clean_labels == K, 0],
            clean_features[clean_labels == K, 1],
            label='Class = {}'.format(K), alpha=.5,
            c=colors[K]
        )
    for K in range(n_classes):
        ax.scatter(
            poisoned_features[poisoned_bases==K, 0],
            poisoned_features[poisoned_bases==K, 1],
            c='k', marker='x'
        )
    ax.legend()
    save_path = os.path.join(save_dir, 'distribution.png')
    plt.savefig(os.path.join(save_path))
    logger.info("Saved feature distributions at {}".format(save_path))
    
def draw_comparison_fig(poison, target, base, filepath):
    # pdb.set_trace()
    _, ax = plt.subplots(ncols=3)
    
    def _reshape(x):
        x = x.permute(1, 2, 0)
        if x.shape[-1] == 1:
            x = x[:, :, 0]
        return x
    
    poison, target, base = _reshape(poison), _reshape(target), _reshape(base)
    ax[0].imshow(base, cmap='gray')
    ax[1].imshow(target, cmap='gray')
    ax[2].imshow(poison, cmap='gray')

    ax[0].set_title("Base")
    ax[1].set_title("Target")
    ax[2].set_title("Poison")

    plt.savefig(filepath)
    logger.info('Saved comparison figure at {}'.format(filepath))

if __name__ == '__main__':
    import model.net as net
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader

    opts = get_opts()
    logger = set_logger(opts.save_dir)
    logger.info('Experiment folder at %s' % opts.save_dir)
    if opts.seed is None:
        logger.warning("SEED NOT SET.")
    else:
        logger.info("Seed: {}".format(opts.seed))

    model = net.MNISTNet(use_lgm=opts.use_lgm).to(opts.device).eval()
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

    n_clean_samples = Y_test.shape[0]
    n_poisoned_samples = int(opts.poisoning_strength * n_clean_samples)
    n_poisoned_samples = opts.max_poisons if (opts.max_poisons > 0) else \
        n_poisoned_samples
    del loader_temp

    poisons, targets, bases = [], [], []

    target_indices = np.random.choice(
        range(n_clean_samples), size=(n_poisoned_samples),
        replace=False
    )

    if opts.debug:
        logger.info("target indices: {}".format(target_indices))
    
    for i in trange(len(target_indices)):
        # select a random target image.
        target_idx = target_indices[i]
        target_img = X_test[target_idx].unsqueeze(0)
        target_label = Y_test[target_idx]

        # select the base image according to the strategy
        if opts.base_strategy == 'random':
            base_label = get_base_class_random(target_label, Y_test)
        elif opts.base_strategy == 'closest':
            base_label = get_base_class_closest(target_label, Y_test,
                                                opts.nn_dict)

        base_img, base_idx = get_random_instance(base_label, X_test, Y_test)
        base_img.unsqueeze_(0)

        logger.info("Crafting Poison")
        logger.info("Target: {}, Base: {}".format(target_label, base_label))
        poison, _ = generate_poison(target_img, base_img, model,
                                     logger, lr=opts.poison_lr,
                                     overlay=opts.overlay,
                                     overlay_alpha=opts.overlay_alpha)
        poisons.append(poison)
        targets.append(target_label.item())
        bases.append(base_label.item())

        poison = poison.squeeze(0)

        # save crafted poison
        filename = '{}_{}_{}.png'.format(base_label, base_idx, i)
        filepath = os.path.join(opts.save_dir, 'poisons',
                                opts.folder_names[target_label],
                                filename)
        save_image(poison, filepath, normalize=True, range=(-1, 1))
        logger.info("Saved image to {}".format(filepath))

        # save comparison matplotlib figure
        filename_fig = '{}_{}_{}_fig.png'.format(base_label, base_idx, i)
        filepath_fig = os.path.join(opts.save_dir, 'poisons',
                                opts.folder_names[target_label],
                                filename_fig)
        # pdb.set_trace()
        draw_comparison_fig(poison.data, target_img.squeeze(0),
                            base_img.squeeze(0), filepath_fig)

    # compute features now for drawing them.
    poisons = torch.cat(poisons)
    targets = np.asarray(targets)
    bases = np.asarray(bases)

    poisons_feats = get_features(poisons, model, logger).cpu().numpy()
    if opts.debug:
        logger.info("Cropping test examples in debug mode.")
        X_test = X_test[:20]
        Y_test = Y_test[:20]
    clean_feats = get_features(X_test, model, logger).cpu().numpy()
    draw_features(clean_feats, Y_test.cpu().numpy(), poisons_feats,
                  bases, targets, opts.save_dir, logger)