import torch
from torch.nn import CrossEntropyLoss as CELoss
from configs.base import Config


class CrossEntropyLoss(CELoss):
    """
    CE Loss an toàn:
    - Nếu model trả (logits, ...), tự lấy phần logits
    - Ép target -> long
    - Hỗ trợ kwargs (weight, label_smoothing, reduction, ...)
    """

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = input[0] if isinstance(input, (tuple, list)) else input
        if target.dtype != torch.long:
            target = target.long()
        return super().forward(logits, target)
