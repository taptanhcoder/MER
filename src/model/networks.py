import torch
import torch.nn as nn

from configs.base import Config
from .modules import build_audio_encoder, build_text_encoder


class MER(nn.Module):
    """
    Multimodal MER cho tiếng Việt:
    - Text: PhoBERT (Roberta) -> last_hidden_state (B, L_t, D_t)
    - Audio: Wav2Vec2 XLSR-53 -> last_hidden_state (B, L_a, D_a)
    - Fusion: Cross-attention hai chiều sau khi chiếu về cùng fusion_dim
    """
    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg

        # ---- Encoders ----
        self.text_encoder = build_text_encoder(cfg.text_encoder_type).to(device)
        for p in self.text_encoder.parameters():
            p.requires_grad = cfg.text_unfreeze

        self.audio_encoder = build_audio_encoder(cfg).to(device)
        for p in self.audio_encoder.parameters():
            p.requires_grad = cfg.audio_unfreeze

        # ---- Project cả 2 về cùng fusion_dim ----
        self.text_proj  = nn.Linear(cfg.text_encoder_dim,  cfg.fusion_dim)
        self.audio_proj = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.text_ln    = nn.LayerNorm(cfg.fusion_dim)
        self.audio_ln   = nn.LayerNorm(cfg.fusion_dim)

        # ---- Cross-Attention (embed_dim = fusion_dim) ----
        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )  # Q = text, K/V = audio

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )  # Q = audio, K/V = text

        # optional post-fusion FFN + LN
        self.post_fusion = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(cfg.fusion_dim),
            nn.Dropout(cfg.dropout),
        )

        self.dropout = nn.Dropout(cfg.dropout)

        # ---- Classification head ----
        self.linear_layer_output = cfg.linear_layer_output
        previous_dim = cfg.fusion_dim
        for i, hidden in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(previous_dim, hidden))
            previous_dim = hidden

        
        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type 

    def forward(
        self,
        input_text,               
        input_audio: torch.Tensor, 
        output_attentions: bool = False,
    ):
        # ---- TEXT ----
        # Hỗ trợ fallback nếu lỡ truyền tensor (không khuyến khích)
        if isinstance(input_text, dict):
            text_out = self.text_encoder(**input_text)               # HF output
        else:
            text_out = self.text_encoder(input_text)                 # not recommended
        text_emb = text_out.last_hidden_state                        # (B, L_t, D_t)
        t = self.text_proj(text_emb)                                 # (B, L_t, D_f)
        t = self.text_ln(t)

        # ---- AUDIO ----
        if input_audio.dim() == 3:                                   # (B, N, T) -> nối theo thời gian
            B, N, T = input_audio.shape
            a_feat = self.audio_encoder(input_audio.view(B * N, T))  # (B*N, L_a, D_a)
            a_feat = a_feat.view(B, N * a_feat.size(1), a_feat.size(2))
        else:                                                        # (B, T)
            a_feat = self.audio_encoder(input_audio)                 # (B, L_a, D_a)

        a = self.audio_proj(a_feat)                                  # (B, L_a, D_f)
        a = self.audio_ln(a)

        # ---- CROSS-ATTENTION (2 chiều) ----
        # Text attends to Audio: Q=t, K/V=a
        text_attention, text_attn_weights = self.text_attention(
            query=t, key=a, value=a, average_attn_weights=False
        )
        text_norm = self.post_fusion(text_attention)
        text_norm = self.dropout(text_norm)

        # Audio attends to Text: Q=a, K/V=t
        audio_attention, audio_attn_weights = self.audio_attention(
            query=a, key=t, value=t, average_attn_weights=False
        )
        audio_norm = self.post_fusion(audio_attention)
        audio_norm = self.dropout(audio_norm)

        # ---- Fusion concat ----
        fusion_norm = torch.cat((text_norm, audio_norm), dim=1)     
        fusion_norm = self.dropout(fusion_norm)

        # ---- Pooling ----
        if self.fusion_head_output_type == "cls":
            # Dùng token đầu của TEXT sau cross-attn (ổn định hơn)
            cls_token_final_fusion_norm = text_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # ---- Classification head ----
        x = self.dropout(cls_token_final_fusion_norm)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [text_attn_weights, audio_attn_weights]

        # Trả thêm tensor như cũ để loss/metrics dùng nếu cần
        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    # ---- tiện ích ----
    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)  # trả (B, L_a, D_a)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state






class TextOnly(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = build_text_encoder(cfg.text_encoder_type).to(device)
        for p in self.text_encoder.parameters():
            p.requires_grad = cfg.text_unfreeze

        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.text_encoder_dim
        for i, hidden in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(previous_dim, hidden))
            previous_dim = hidden
        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(self, input_text, input_audio=None, output_attentions: bool = False):
        if isinstance(input_text, dict):
            text_emb = self.text_encoder(**input_text).last_hidden_state
        else:
            text_emb = self.text_encoder(input_text).last_hidden_state

        fusion_norm = self.dropout(text_emb)

        if self.fusion_head_output_type == "cls":
            pooled = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            pooled = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            pooled = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            pooled = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)

        return out, pooled


class AudioOnly(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = build_audio_encoder(cfg).to(device)
        for p in self.audio_encoder.parameters():
            p.requires_grad = cfg.audio_unfreeze

        self.linear_layer_output = cfg.linear_layer_output
        self.dropout = nn.Dropout(cfg.dropout)

        previous_dim = cfg.audio_encoder_dim
        for i, hidden in enumerate(self.linear_layer_output):
            setattr(self, f"linear_{i}", nn.Linear(previous_dim, hidden))
            previous_dim = hidden
        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(self, input_text=None, input_audio: torch.Tensor = None, output_attentions: bool = False):
        if input_audio.dim() == 3:
            B, N, T = input_audio.shape
            a_feat = self.audio_encoder(input_audio.view(B * N, T))    # (B*N, L_a, D_a)
            a_feat = a_feat.view(B, N * a_feat.size(1), a_feat.size(2))
        else:
            a_feat = self.audio_encoder(input_audio)                   # (B, L_a, D_a)

        fusion_norm = self.dropout(a_feat)

        if self.fusion_head_output_type == "cls":
            pooled = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            pooled = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            pooled = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            pooled = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)

        return out, pooled
