import torch
import numpy as np

def compute_loss(model, curr_poison, base_img, target_img, beta_zero):
    with torch.no_grad():
        a = torch.norm(model(curr_poison) - model(target_img))
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
    beta_zero = beta * (2048.0 / base_img.numel) ** 2
    Lp_func = lambda x: torch.norm(model(x)[1] - model(target_img)[1])

    poison = base_img.clone()
    loss = compute_loss(model, poison, base_img, target_img, beta_zero)
    
    for _ in range(max_iters):
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
            break
        
        if new_loss > loss: # loss is too big than before; don't update stuff.
            lr *= decay_coeff
        else: # loss is lower than before and life is good; update stuff.
            poison, loss = new_poison, new_loss
    return poison, loss

# testing script
if __name__ == '__main__':
    pass