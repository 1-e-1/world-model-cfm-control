"""Shared attention primitive used by the retained V7 spatiotemporal model.

Layer names are unchanged so existing state_dict checkpoints load strictly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, source: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, query_count, dim = query.shape
        source_count = source.shape[1]
        q = self.q(query).view(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(source).view(batch, source_count, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(source).view(batch, source_count, self.heads, self.head_dim).transpose(1, 2)
        attention_mask = mask[:, None, None, :] if mask is not None else None
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.out(result.transpose(1, 2).reshape(batch, query_count, dim))
