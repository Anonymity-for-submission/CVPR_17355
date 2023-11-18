import torch
     
def align_loss(x, y, alpha=2):
    # print(x.shape)
    # print(y.shape)
    return (x - y).norm(p=2, dim=0).pow(alpha).mean()

def uniform_loss(x, t=2):
    return -torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()