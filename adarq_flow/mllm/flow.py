"""Flow-matching (rectified-flow / conditional-OT) utilities for the residual head.

Replaces Bifrost-1's MSE on continuous latents with a proper distributional
objective. The path is the straight (rectified) interpolation
``x_t = (1 - t) * x0 + t * x1`` with constant velocity target ``u = x1 - x0``,
where ``x1`` is the data (the continuous residual) and ``x0 ~ N(0, I)``.

``velocity_fn(x_t, t) -> v`` closes over any conditioning (hidden state, prefix).
No diffusion anywhere.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def rectified_flow_target(x1: Tensor, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(x_t, u)`` for the straight path. ``t`` is ``[M]`` in ``[0, 1]``."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0/x1 shape mismatch: {tuple(x0.shape)} vs {tuple(x1.shape)}")
    tt = t.view(-1, *([1] * (x1.dim() - 1)))
    x_t = (1.0 - tt) * x0 + tt * x1
    u = x1 - x0
    return x_t, u


def flow_matching_loss(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    x1: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Mean squared velocity error over one random ``(t, x0)`` draw per sample."""
    if x1.numel() == 0:
        raise ValueError("flow_matching_loss received an empty target")
    x0 = torch.empty_like(x1).normal_(generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    x_t, u = rectified_flow_target(x1, x0, t)
    v = velocity_fn(x_t, t)
    if v.shape != u.shape:
        raise ValueError(f"velocity shape {tuple(v.shape)} != target {tuple(u.shape)}")
    return (v - u).pow(2).sum(-1).mean()


@torch.no_grad()
def flow_sample(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    num: int,
    dim: int,
    steps: int,
    device: torch.device,
    x0: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Integrate the ODE ``dx/dt = v`` from ``t=0`` (noise) to ``t=1`` (data) by Euler."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if x0 is None:
        x0 = torch.empty(num, dim, device=device).normal_(generator=generator)
    elif x0.shape != (num, dim):
        raise ValueError(f"x0 shape {tuple(x0.shape)} != ({num}, {dim})")
    x = x0
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((num,), s * dt, device=device)
        x = x + dt * velocity_fn(x, t)
    return x
