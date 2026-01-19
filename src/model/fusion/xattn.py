import torch
import torch.nn as nn
from configs.base import Config
from .base import FusionStrategy


class FusionCrossAttention(FusionStrategy):

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.post_fusion = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(cfg.fusion_dim),
            nn.Dropout(cfg.dropout),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, t, a, kpm_text=None, kpm_audio=None):
 
        text_attention, w_t = self.text_attention(
            query=t, key=a, value=a, key_padding_mask=kpm_audio, average_attn_weights=False
        )
        text_norm = self.dropout(self.post_fusion(text_attention))

        audio_attention, w_a = self.audio_attention(
            query=a, key=t, value=t, key_padding_mask=kpm_text, average_attn_weights=False
        )
        audio_norm = self.dropout(self.post_fusion(audio_attention))

        fusion_seq = torch.cat((text_norm, audio_norm), dim=1)
        return fusion_seq, None, {"text_attn": w_t, "audio_attn": w_a}
