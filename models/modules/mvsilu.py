import torch
from torch import nn

from .utils import unsqueeze_like
from algebra.norms import calculate_norms, qs


class MVSiLU(nn.Module):
    def __init__(self, algebra, metric, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.exclude_dual = exclude_dual
        self.invariant = invariant
        self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))
        self.metric = metric

        if invariant == "norm":
            self._get_invariants = self._norms_except_scalar
        elif invariant == "mag2":
            self._get_invariants = self._mag2s_except_scalar
        else:
            raise ValueError(f"Invariant {invariant} not recognized.")

    def _norms_except_scalar(self, input):
        return calculate_norms(self.algebra, torch.linalg.eig(self.metric)[0].real, input, grades=self.algebra.grades[1:].to('cuda'))

    def _mag2s_except_scalar(self, input):
        #return qs(self.algebra, self.metric, input, grades=self.algebra.grades[1:].to('cuda'))
        # tensor([0.8778, 0.5711, 0.3511], device='cuda:0', grad_fn=<SelectBackward0>)
        return qs(self.algebra, torch.linalg.eig(self.metric)[0].real, input, grades=self.algebra.grades[1:].to('cuda'))

    def forward(self, input):
        norms = self._get_invariants(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.sigmoid(norms) * input



class MVReLU(nn.Module):
    def __init__(self, algebra, metric, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.exclude_dual = exclude_dual
        self.invariant = invariant
        self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))
        self.metric = metric

        if invariant == "norm":
            self._get_invariants = self._norms_except_scalar
        elif invariant == "mag2":
            self._get_invariants = self._mag2s_except_scalar
        else:
            raise ValueError(f"Invariant {invariant} not recognized.")

    def _norms_except_scalar(self, input):
        # Computes norms of input elements, excluding the scalar part
        return calculate_norms(self.algebra, torch.linalg.eig(self.metric)[0].real, input, grades=self.algebra.grades[1:].to('cuda'))

    def _mag2s_except_scalar(self, input):
        return qs(self.algebra, torch.linalg.eig(self.metric)[0].real, input, grades=self.algebra.grades[1:].to('cuda'))

    def forward(self, input):
        norms = self._get_invariants(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.relu(norms) * input
