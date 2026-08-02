"""The trainable vision-generation branch.

A stack of :class:`BranchBlock`s: Img-G patch tokens attend bidirectionally over
themselves and (read-only) over the frozen context. On real configs this is
initialized from a copy of the MLLM's QKV/MLP/norm layers (Phase-2 follow-up);
here it is a freshly-initialized trainable transformer of matching width/depth.
"""

from __future__ import annotations

from torch import Tensor, nn

from .layers import BranchBlock


class VisionGenBranch(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.blocks = nn.ModuleList(
            BranchBlock(hidden_dim, num_heads) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imgg: Tensor, context: Tensor) -> Tensor:
        """``imgg`` ``[B, N, H]`` queries; ``context`` ``[B, T, H]`` frozen key/values."""
        if imgg.dim() != 3 or context.dim() != 3:
            raise ValueError("imgg and context must be [B, *, H]")
        if imgg.shape[0] != context.shape[0] or imgg.shape[-1] != context.shape[-1]:
            raise ValueError(
                f"batch/width mismatch: imgg {tuple(imgg.shape)} context {tuple(context.shape)}"
            )
        x = imgg
        for blk in self.blocks:
            x = blk(x, context)
        return self.norm(x)
