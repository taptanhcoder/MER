
import torch
import torch.nn as nn

from configs.base import Config
from model.audio import build_audio_encoder
from model.text import build_text_encoder
from model.fusion import (
    AttnPool,
    masked_reduce,
    FusionCrossAttention,
    FusionBiLstmAttention,
    FusionCnnBiLstmAttention,
)


class MER(nn.Module):

    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg

        # ---- Text encoder ----
        self.text_encoder = build_text_encoder(cfg.text_encoder_type, cfg.text_encoder_ckpt).to(device)
        if hasattr(self.text_encoder, "config") and hasattr(self.text_encoder.config, "hidden_size"):
            cfg.text_encoder_dim = int(self.text_encoder.config.hidden_size)
        for p in self.text_encoder.parameters():
            p.requires_grad = cfg.text_unfreeze

        # ---- Audio encoder ----
        self.audio_encoder = build_audio_encoder(cfg).to(device)
        if hasattr(self.audio_encoder, "config") and hasattr(self.audio_encoder.config, "hidden_size"):
            cfg.audio_encoder_dim = int(self.audio_encoder.config.hidden_size)
        for p in self.audio_encoder.parameters():
            p.requires_grad = cfg.audio_unfreeze

        # ---- Project về fusion_dim ----
        self.text_proj  = nn.Linear(cfg.text_encoder_dim,  cfg.fusion_dim)
        self.audio_proj = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.text_ln    = nn.LayerNorm(cfg.fusion_dim)
        self.audio_ln   = nn.LayerNorm(cfg.fusion_dim)
        self.dropout    = nn.Dropout(cfg.dropout)

        # ---- Fusion strategy ----
        ftype = (cfg.fusion_type or "xattn").lower()
        if ftype == "bilstm_attn":
            self.fusion = FusionBiLstmAttention(cfg)
        elif ftype == "cnn_bilstm_attn":
            self.fusion = FusionCnnBiLstmAttention(cfg)
        else:
            self.fusion = FusionCrossAttention(cfg)

        # ---- Classification head ----
        self.linear_layer_output = cfg.linear_layer_output
        previous_dim = cfg.fusion_dim
        if cfg.fusion_head_output_type == "attn" and int(getattr(cfg, "fusion_pool_heads", 1)) > 1:
            previous_dim = cfg.fusion_dim * int(cfg.fusion_pool_heads)

        for i, hidden in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(previous_dim, hidden))
            previous_dim = hidden

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.classifier = self.classifer  # alias
        self.fusion_head_output_type = cfg.fusion_head_output_type

    @staticmethod
    def _build_text_kpm(input_text) -> torch.Tensor | None:
        if isinstance(input_text, dict) and ("attention_mask" in input_text):
            return (input_text["attention_mask"] == 0)
        return None

    @staticmethod
    def _build_audio_attn_mask(input_audio: torch.Tensor, meta: dict, device) -> torch.Tensor:
        B, T = input_audio.shape
        if meta is not None and "audio_attn_mask" in meta:
            return meta["audio_attn_mask"].to(device)
        if meta is not None and "audio_lengths" in meta:
            lengths = torch.as_tensor(meta["audio_lengths"], device=device, dtype=torch.long)
            mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).long()
            return mask
        return torch.ones((B, T), device=device, dtype=torch.long)

    @torch.no_grad()
    def _build_audio_kpm_after_encoder(self, a_seq: torch.Tensor, meta: dict, device) -> torch.Tensor | None:
        if meta is None or ("audio_lengths" not in meta):
            return None
        if not hasattr(self.audio_encoder, "get_feat_lengths"):
            return None
        lengths_wav = torch.as_tensor(meta["audio_lengths"], device=device, dtype=torch.long)
        L_valid = self.audio_encoder.get_feat_lengths(lengths_wav)
        L_enc = a_seq.size(1)
        L_valid = torch.clamp(L_valid, min=1, max=L_enc)
        kpm = (torch.arange(L_enc, device=device).unsqueeze(0) >= L_valid.unsqueeze(1))
        all_mask = kpm.all(dim=1)
        if bool(all_mask.any().item()):
            kpm[all_mask] = False
        return kpm

    def forward(self, input_text, input_audio: torch.Tensor, meta: dict = None, output_attentions: bool = False):
        device = input_audio.device

        # ---- TEXT ----
        if isinstance(input_text, dict):
            text_out = self.text_encoder(**input_text)
        else:
            text_out = self.text_encoder(input_text)
        text_emb = text_out.last_hidden_state
        t = self.text_ln(self.text_proj(text_emb))
        kpm_text = self._build_text_kpm(input_text)

        # ---- AUDIO (trước encoder) ----
        attn_mask_audio_input = self._build_audio_attn_mask(input_audio, meta, device)

        # ---- AUDIO (encoder) ----
        a_feat = self.audio_encoder(input_audio, attention_mask=attn_mask_audio_input)
        a = self.audio_ln(self.audio_proj(a_feat))

        # ---- AUDIO key_padding_mask sau encoder ----
        kpm_audio = self._build_audio_kpm_after_encoder(a, meta, device)

        # ---- FUSION ----
        fusion_seq, pooled_from_fusion, aux = self.fusion(t, a, kpm_text=kpm_text, kpm_audio=kpm_audio)

        # ---- POOLING ----
        if pooled_from_fusion is not None and self.fusion_head_output_type == "attn":
            pooled = pooled_from_fusion
        else:
            if (kpm_text is not None) and (kpm_audio is not None):
                kpm_fusion = torch.cat([kpm_text, kpm_audio], dim=1)
            elif kpm_text is not None:
                kpm_fusion = torch.cat(
                    [kpm_text, torch.zeros(fusion_seq.size(0), a.size(1), device=fusion_seq.device, dtype=torch.bool)],
                    dim=1
                )
            elif kpm_audio is not None:
                kpm_fusion = torch.cat(
                    [torch.zeros(fusion_seq.size(0), t.size(1), device=fusion_seq.device, dtype=torch.bool), kpm_audio],
                    dim=1
                )
            else:
                kpm_fusion = None

            if self.fusion_head_output_type == "attn":
                pooled = AttnPool(self.cfg.fusion_dim, num_heads=1, dropout=self.cfg.dropout)(fusion_seq, kpm=kpm_fusion)
            else:
                pooled = masked_reduce(fusion_seq, kpm_fusion, self.fusion_head_output_type)

        # ---- Classification head ----
        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, pooled], aux
        return out, pooled, None, None


class TextOnly(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = build_text_encoder(cfg.text_encoder_type, cfg.text_encoder_ckpt).to(device)
        if hasattr(self.text_encoder, "config") and hasattr(self.text_encoder.config, "hidden_size"):
            cfg.text_encoder_dim = int(self.text_encoder.config.hidden_size)
        for p in self.text_encoder.parameters():
            p.requires_grad = cfg.text_unfreeze

        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.text_encoder_dim
        for i, hidden in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(previous_dim, hidden))
            previous_dim = hidden

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.classifier = self.classifer
        self.fusion_head_output_type = cfg.fusion_head_output_type
        self.attn_pool_text = AttnPool(d_model=cfg.text_encoder_dim, num_heads=1, dropout=cfg.dropout)

    @staticmethod
    def _build_text_kpm(input_text) -> torch.Tensor | None:
        if isinstance(input_text, dict) and ("attention_mask" in input_text):
            return (input_text["attention_mask"] == 0)
        return None

    def forward(self, input_text, input_audio=None, output_attentions: bool = False):
        if isinstance(input_text, dict):
            enc_out = self.text_encoder(**input_text)
        else:
            enc_out = self.text_encoder(input_text)
        text_emb = enc_out.last_hidden_state
        kpm_text = self._build_text_kpm(input_text)

        seq = self.dropout(text_emb)
        if self.fusion_head_output_type == "cls":
            pooled = seq[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            if kpm_text is None:
                pooled = seq.mean(dim=1)
            else:
                valid = ~kpm_text
                denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
                pooled = (seq * valid.unsqueeze(-1)).sum(dim=1) / denom
        elif self.fusion_head_output_type == "max":
            if kpm_text is None:
                pooled = seq.max(dim=1)[0]
            else:
                masked = seq.masked_fill(kpm_text.unsqueeze(-1), float("-inf"))
                pooled = masked.max(dim=1)[0]
                bad = torch.isinf(pooled).any(dim=1)
                if bool(bad.any().item()):
                    pooled[bad] = seq[bad, 0, :]
        elif self.fusion_head_output_type == "min":
            if kpm_text is None:
                pooled = seq.min(dim=1)[0]
            else:
                masked = seq.masked_fill(kpm_text.unsqueeze(-1), float("+inf"))
                pooled = masked.min(dim=1)[0]
                bad = torch.isinf(pooled).any(dim=1)
                if bool(bad.any().item()):
                    pooled[bad] = seq[bad, 0, :]
        elif self.fusion_head_output_type == "attn":
            pooled = self.attn_pool_text(seq, kpm=kpm_text)
        else:
            raise ValueError("Invalid fusion head output type")

        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)
        return out, pooled


def build_text_only(cfg: Config, device: str = "cpu") -> TextOnly:
    return TextOnly(cfg, device=device)


class AudioOnly(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = build_audio_encoder(cfg).to(device)
        if hasattr(self.audio_encoder, "config") and hasattr(self.audio_encoder.config, "hidden_size"):
            cfg.audio_encoder_dim = int(self.audio_encoder.config.hidden_size)
        for p in self.audio_encoder.parameters():
            p.requires_grad = cfg.audio_unfreeze

        self.linear_layer_output = cfg.linear_layer_output
        self.dropout = nn.Dropout(cfg.dropout)

        previous_dim = cfg.audio_encoder_dim
        for i, hidden in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(previous_dim, hidden))
            previous_dim = hidden

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.classifier = self.classifer
        self.fusion_head_output_type = cfg.fusion_head_output_type
        self.attn_pool_audio = AttnPool(d_model=cfg.audio_encoder_dim, num_heads=1, dropout=cfg.dropout)

    @staticmethod
    def _build_audio_attn_mask(input_audio: torch.Tensor, meta: dict, device) -> torch.Tensor:
        B, T = input_audio.shape
        if meta is not None and "audio_attn_mask" in meta:
            return meta["audio_attn_mask"].to(device)
        if meta is not None and "audio_lengths" in meta:
            lengths = torch.as_tensor(meta["audio_lengths"], device=device, dtype=torch.long)
            mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).long()
            return mask
        return torch.ones((B, T), device=device, dtype=torch.long)

    @torch.no_grad()
    def _build_audio_kpm_after_encoder(self, a_seq: torch.Tensor, meta: dict, device) -> torch.Tensor | None:
        if meta is None or ("audio_lengths" not in meta):
            return None
        if not hasattr(self.audio_encoder, "get_feat_lengths"):
            return None
        lengths = torch.as_tensor(meta["audio_lengths"], device=device, dtype=torch.long)
        L_valid = self.audio_encoder.get_feat_lengths(lengths)
        L_enc = a_seq.size(1)
        L_valid = torch.clamp(L_valid, min=1, max=L_enc)
        kpm = (torch.arange(L_enc, device=device).unsqueeze(0) >= L_valid.unsqueeze(1))
        all_mask = kpm.all(dim=1)
        if bool(all_mask.any().item()):
            kpm[all_mask] = False
        return kpm

    def forward(self, input_text=None, input_audio: torch.Tensor = None, meta: dict = None, output_attentions: bool = False):
        if input_audio.dim() == 3:
            B0, N, T = input_audio.shape
            input_audio = input_audio.view(B0 * N, T)
            batched_windows = True
        else:
            B0, T = input_audio.shape
            batched_windows = False

        device = input_audio.device
        attn_mask_audio_input = self._build_audio_attn_mask(input_audio, meta, device)
        a_feat = self.audio_encoder(input_audio, attention_mask=attn_mask_audio_input)
        kpm_audio = self._build_audio_kpm_after_encoder(a_feat, meta, device)

        if batched_windows:
            a_feat = a_feat.view(B0, -1, a_feat.size(2))
            if kpm_audio is not None:
                kpm_audio = kpm_audio.view(B0, -1)

        if self.fusion_head_output_type == "attn":
            pooled = self.attn_pool_audio(a_feat, kpm=kpm_audio)
        else:
            pooled = masked_reduce(a_feat, kpm_audio, self.fusion_head_output_type)

        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)
        return out, pooled


def build_audio_only(cfg: Config, device: str = "cpu") -> AudioOnly:
    return AudioOnly(cfg, device=device)


MemoCMT = MER
