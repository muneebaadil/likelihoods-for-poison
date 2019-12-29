import torch
import numpy as np

def compute_loss(model, curr_poison, base_img, target_img, beta_zero):
    with torch.no_grad():
        a = torch.norm(model(curr_poison)[0] - \
            model(target_img)[0])
        b = torch.norm(curr_poison - base_img)
        out = a + beta_zero * b
    return out

def do_poisoning(target_img, base_img, model, beta=0.25, max_iters=1000,
                 loss_thres=2.9, lr=500.*255, decay_coeff=.5, min_val=-1.,
                 max_val=1.):
    """
    Does poisoning according to Poison Frogs paper.
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

    print("Initial loss = {}".format(loss))
    
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
                print("Optimization done: Loss = {}".format(loss))
                break
            
            if new_loss > loss: # loss is too big than before; don't update stuff.
                lr *= decay_coeff
            else: # loss is lower than before and life is good; update stuff.
                poison, loss = new_poison, new_loss
    print("Final loss = {}".format(loss))
    return poison, loss


# testing script
if __name__ == '__main__':
    import model.net as net
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model = net.Net()
    model.load_state_dict(torch.load('../experiments/2019-12-27_14-41-06/models/epoch-best.model',
                          map_location='cpu'))
    t = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)))
    )
    data = datasets.MNIST(
        root='../datasets/', train=False, download=False, transform=t
    )
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=False)
    
    X, Y = [], []
    print("Loading dataset into the memory")
    for (X_, Y_) in tqdm(loader):
        X.append(X_)
        Y.append(Y_)
    X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
    target_img = X[Y == 4][0] # selecting the first image with label 4.
    base_img = X[Y == 1][0] # selecting the first iamge with label 1.

    del X
    poison_label = torch.tensor([10], dtype=torch.long)
    Y = torch.cat((Y, poison_label), 0)

    print("Crafting poison...")
    poison, loss = do_poisoning(target_img.unsqueeze(0), 
                 base_img.unsqueeze(0), model)
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=3)

    poison_ = poison.squeeze(0).squeeze(0)
    target_img_ = target_img.squeeze(0)
    base_img_ = base_img.squeeze(0)

    ax[0].imshow(poison_.detach().numpy(), cmap='gray')
    ax[1].imshow(target_img_.numpy(), cmap='gray')
    ax[2].imshow(base_img_.numpy(), cmap='gray')
    # plt.show()
    plt.savefig('poisoning.png')

    print("Plotting Distribution")
    feats = []
    for (X, _) in tqdm(loader):
        _, feat = model(X)
        feats.append(feat)
    poison_feat = model(poison)
    feats.append(poison_feat)

    feats = torch.cat(feats, dim=0)
    fig, ax = plt.subplots()
    for K in range(11):
        ax.scatter(feats[Y == K, 0], feats_all[Y == K, 1],
                    label='Class = {}'.format(K))

    ax.legend()
    # plt.show()
    plt.save_fig('poisoning_dist.png') 