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
    
    for _ in tqdm(range(max_iters)):
        # calculate gradient of Lp w.r.t. x
        poison.requires_grad = True
        norm_val = Lp_func(poison)

        model.zero_grad()
        norm_val.backward()
        grad_Lp = poison.grad.data

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
    return poison, loss


# testing script
if __name__ == '__main__':
    import model.net2 as net
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model = net.Net()
    model.load_state_dict(torch.load('../testdrivelgm.epoch-39-.model',
                          map_location='cpu'))
    # TODO: LOAD THE MODEL WEIGHTS HERE.
    # model.load_state_dict(state_dict=torch.load("../experiments/"))

    t = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)))
    )
    data = datasets.MNIST(
        root='../datasets/', train=False, download=False, transform=t
    )
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=False)
    
    X, Y = [], []
    print("Loading dataset into the memory")
    for (X_, Y_) in tqdm(loader):
        X.append(X_)
        Y.append(Y_)
    
    X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
    target_img = X[Y == 4][0] # selecting the first image with label 4.
    base_img = X[Y == 1][0] # selecting the first iamge with label 1.

    
    print("Creating poison...")
    poison, loss = do_poisoning(target_img.unsqueeze(0), 
                 base_img.unsqueeze(0), model)
    
    
    import matplotlib.pyplot as plt
    # import pdb
    # pdb.set_trace()
    poison_ = poison.squeeze(0).permute(1,2,0)
    plt.imshow(poison.detach().numpy(), cmap='gray')