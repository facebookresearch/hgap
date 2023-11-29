import torch

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.expm1(torch.abs(x)))
