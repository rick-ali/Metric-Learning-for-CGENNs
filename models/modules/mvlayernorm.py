import torch
from torch import nn

from models.modules.utils import unsqueeze_like

from algebra.norms import calculate_norm

EPS = 1e-6


class MVLayerNorm(nn.Module):
    def __init__(self, algebra, metric, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))
        self.metric = metric

    def forward(self, input):
        norm = calculate_norm(self.algebra, torch.linalg.eig(self.metric)[0].real, input)[..., :1].mean(dim=1, keepdim=True) + EPS
        a = unsqueeze_like(self.a, norm, dim=2)
        return a * input / norm



class MVBatchNorm(nn.Module):
    def __init__(self, algebra, metric, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))
        self.metric = metric

    def forward(self, input):

        norm = calculate_norm(self.algebra, self.metric, input)[..., :1].mean(dim=0, keepdim=True) + EPS
        a = unsqueeze_like(self.a, norm, dim=2)
        return a * input / norm

    
