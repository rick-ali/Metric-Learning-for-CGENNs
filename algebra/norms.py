import functools
import math

import torch
from torch import nn

from .metric import ShortLexBasisBladeOrder, construct_gmt, gmt_element



def beta(algebra, mv, blades=None):
    signs = algebra._beta_signs
    if blades is not None:
        signs = signs[blades]
    return signs * mv.clone()


def b(algebra, metric, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            beta_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(algebra.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            beta_blades = None

        return geometric_product(
            algebra,
            metric,
            beta(algebra, x, blades=beta_blades),
            y,
            blades=blades,
        )

def q(algebra, metric, mv, blades=None):
    if blades is not None:
        blades = (blades, blades)
    return b(algebra, metric, mv, mv, blades=blades)

def _smooth_abs_sqrt(input, eps=1e-16):
    return (input**2 + eps) ** 0.25

def calculate_norm(algebra, metric, mv, blades=None):
    return _smooth_abs_sqrt(q(algebra, metric, mv, blades=blades))


def calculate_norms(algebra, metric, mv, grades=None):
    if grades is None:
        grades = algebra.grades
    return [
        calculate_norm(algebra, metric, algebra.get_grade(mv, grade), blades=algebra.grade_to_index[grade])
        for grade in grades
    ]

def qs(algebra, metric, mv, grades=None):
    if grades is None:
        grades = algebra.grades
    return [
        q(algebra, metric, algebra.get_grade(mv, grade), blades=algebra.grade_to_index[grade])
        for grade in grades
        ]


def geometric_product(algebra, metric, a, b, blades=None):
        cayley = (
            construct_gmt(
                algebra.bbo.index_to_bitmap, algebra.bbo.bitmap_to_index, metric
                #self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric
            )
            .to_dense()
            .to(torch.get_default_dtype())
        ).to('cuda')

        if blades is not None:
            blades_l, blades_o, blades_r = blades
            assert isinstance(blades_l, torch.Tensor)
            assert isinstance(blades_o, torch.Tensor)
            assert isinstance(blades_r, torch.Tensor)
            cayley = cayley[blades_l[:, None, None].to('cuda'), blades_o[:, None].to('cuda'), blades_r.to('cuda')]

        return torch.einsum("...i,ijk,...k->...j", a, cayley, b)

def cayley(algebra, metric):
    return construct_gmt(algebra.bbo.index_to_bitmap.to('cuda'), 
                         algebra.bbo.bitmap_to_index.to('cuda'), 
                         metric).to_dense().to(torch.get_default_dtype()).to('cuda')