import torch.nn as nn
from configs.base import Config


class FusionStrategy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, t, a, kpm_text=None, kpm_audio=None):
        raise NotImplementedError
