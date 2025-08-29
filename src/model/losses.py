import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as _CELoss
from configs.base import Config


class CrossEntropyLoss(_CELoss):
    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = input[0] if isinstance(input, (tuple, list)) else input
        if target.dtype != torch.long:
            target = target.long()
        return super().forward(logits, target)


class FocalLoss(torch.nn.Module):
    def __init__(self, cfg: Config = None, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(getattr(cfg, "loss_gamma", gamma)) if cfg is not None else gamma
        self.reduction = reduction

    def forward(self, input, target):
        logits = input[0] if isinstance(input, (tuple, list)) else input
        if target.dtype != torch.long:
            target = target.long()
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss


class LabelSmoothingCE(torch.nn.Module):
    def __init__(self, cfg: Config = None, smoothing: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.smoothing = float(getattr(cfg, "label_smoothing", smoothing)) if cfg is not None else smoothing
        self.reduction = reduction

    def forward(self, input, target):
        logits = input[0] if isinstance(input, (tuple, list)) else input
        if target.dtype != torch.long:
            target = target.long()
        n_class = logits.size(-1)
        logprobs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-true_dist * logprobs).sum(dim=-1)
        return loss.mean() if self.reduction == "mean" else loss


def get_loss(cfg: Config):
    lt = getattr(cfg, "loss_type", "CrossEntropyLoss").lower()
    if lt == "crossentropyloss":
        return CrossEntropyLoss(cfg)
    if lt == "focalloss":
        return FocalLoss(cfg)
    if lt in ("labelsmoothingce", "labelsmoothing"):
        return LabelSmoothingCE(cfg)
    raise ValueError(f"Unsupported loss_type: {cfg.loss_type}")
