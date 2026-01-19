import torch
import torch.nn as nn


class AttnPool(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_heads = max(1, int(num_heads))
        self.q = nn.Parameter(torch.randn(self.num_heads, d_model))
        self.w = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kpm: torch.Tensor | None = None) -> torch.Tensor:
        B, L, F = x.shape
        keys = self.w(x)


        scores = torch.einsum("hf,blf->hbl", self.q, keys)
        if kpm is not None:
            scores = scores.masked_fill(kpm.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        pooled = torch.einsum("hbl,blf->hbf", attn, x).transpose(0, 1).contiguous()
        if self.num_heads == 1:
            return pooled[:, 0, :]
        return pooled.reshape(B, self.num_heads * F)


def masked_reduce(x: torch.Tensor, kpm: torch.Tensor | None, mode: str) -> torch.Tensor:

    if mode == "cls":
        return x[:, 0, :]

    if kpm is None:
        if mode == "mean": return x.mean(dim=1)
        if mode == "max":  return x.max(dim=1)[0]
        if mode == "min":  return x.min(dim=1)[0]
        raise ValueError("Invalid fusion head output type")

    valid = ~kpm
    if mode == "mean":
        denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

    if mode == "max":
        masked = x.masked_fill(kpm.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1)[0]
        bad = torch.isinf(pooled).any(dim=1)
        if bool(bad.any().item()):
            pooled[bad] = x[bad, 0, :]
        return pooled

    if mode == "min":
        masked = x.masked_fill(kpm.unsqueeze(-1), float("+inf"))
        pooled = masked.min(dim=1)[0]
        bad = torch.isinf(pooled).any(dim=1)
        if bool(bad.any().item()):
            pooled[bad] = x[bad, 0, :]
        return pooled

    raise ValueError("Invalid fusion head output type")
