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
        )  

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        ) 

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

    def _build_masks(self, input_text, input_audio, meta, device):

        B, T = input_audio.shape

        if meta is not None and "audio_lengths" in meta:
            lengths = torch.as_tensor(meta["audio_lengths"], device=device, dtype=torch.long)
            attn_mask_audio_input = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).long()
        else:
            attn_mask_audio_input = torch.ones((B, T), device=device, dtype=torch.long)


        if isinstance(input_text, dict) and "attention_mask" in input_text:
            kpm_text = (input_text["attention_mask"] == 0)  
        else:
            kpm_text = None

        return attn_mask_audio_input, kpm_text

    def forward(
        self,
        input_text,              
        input_audio: torch.Tensor,   
        meta: dict = None,
        output_attentions: bool = False,
    ):
        device = input_audio.device

        # ---- TEXT ----
        if isinstance(input_text, dict):
            text_out = self.text_encoder(**input_text)              
            t_mask = input_text.get("attention_mask", None)
        else:
            text_out = self.text_encoder(input_text)
            t_mask = None
        text_emb = text_out.last_hidden_state                       
        t = self.text_proj(text_emb)                               
        t = self.text_ln(t)

        # ---- AUDIO ----
        attn_mask_audio_input, kpm_text = self._build_masks(input_text, input_audio, meta, device)
        a_feat = self.audio_encoder(input_audio, attention_mask=attn_mask_audio_input) 
        a = self.audio_proj(a_feat)                                
        a = self.audio_ln(a)

        if meta is not None and "audio_lengths" in meta:
            lengths = torch.as_tensor(meta["audio_lengths"], device=device, dtype=torch.long)
            L_valid = self.audio_encoder.get_feat_lengths(lengths)  
            L_a = a.size(1)
    
            L_valid = torch.clamp(L_valid, min=1, max=L_a)
            kpm_audio = (torch.arange(L_a, device=device).unsqueeze(0) >= L_valid.unsqueeze(1))

            all_mask = kpm_audio.all(dim=1)
            if bool(all_mask.any().item()):
                kpm_audio[all_mask] = False
        else:
            kpm_audio = None

        # ---- CROSS-ATTENTION (2 chiều) ----

        text_attention, text_attn_weights = self.text_attention(
            query=t, key=a, value=a, key_padding_mask=kpm_audio, average_attn_weights=False
        )
        text_norm = self.post_fusion(text_attention)
        text_norm = self.dropout(text_norm)


        audio_attention, audio_attn_weights = self.audio_attention(
            query=a, key=t, value=t, key_padding_mask=kpm_text, average_attn_weights=False
        )
        audio_norm = self.post_fusion(audio_attention)
        audio_norm = self.dropout(audio_norm)

        # ---- Fusion concat ----
        fusion_norm = torch.cat((text_norm, audio_norm), dim=1)
        fusion_norm = self.dropout(fusion_norm)

        # ---- Pooling ----
        if self.fusion_head_output_type == "cls":
            pooled = text_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            pooled = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            pooled = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            pooled = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # ---- Classification head ----
        x = self.dropout(pooled)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, pooled], [text_attn_weights, audio_attn_weights]

        return out, pooled, text_norm, audio_norm


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
            a_feat = self.audio_encoder(input_audio.view(B * N, T))
            a_feat = a_feat.view(B, N * a_feat.size(1), a_feat.size(2))
        else:
            a_feat = self.audio_encoder(input_audio)

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



