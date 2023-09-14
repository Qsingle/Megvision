from typing import Type

import megengine as mge
import megengine.functional as F
import megengine.module as M


class MLPBlock(M.Module):
    def __init__(
        self, embedding_dim: int, mlp_dim: int, act: Type[M.Module] = M.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = M.Linear(embedding_dim, mlp_dim)
        self.lin2 = M.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(M.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mge.Parameter(F.ones(num_channels))
        self.bias = mge.Parameter(F.zeros(num_channels))
        self.eps = eps

    def forward(self, x: mge.Tensor) -> mge.Tensor:
        u = x.mean(1, keepdims=True)
        s = F.pow((x - u), 2).mean(1, keepdims=True)
        x = (x - u) / F.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
