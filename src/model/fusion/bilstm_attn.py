import torch
import torch.nn as nn
from configs.base import Config
from .base import FusionStrategy
from .pooling import AttnPool


class _FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _CrossBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = _FFN(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, kpm_kv=None):
        q_ln = self.ln_q(q)
        out, w = self.attn(q_ln, kv, kv, key_padding_mask=kpm_kv, average_attn_weights=False)
        q = q + self.drop(out)
        q_ff = self.ln_ff(q)
        q = q + self.ffn(q_ff)
        return q, w


class FusionBiLstmAttention(FusionStrategy):

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        F = cfg.fusion_dim


        self.text_lstm = nn.LSTM(
            input_size=F,
            hidden_size=cfg.fusion_bilstm_hidden_text,
            num_layers=cfg.fusion_bilstm_layers,
            dropout=cfg.fusion_bilstm_dropout if cfg.fusion_bilstm_layers > 1 else 0.0,
            bidirectional=cfg.fusion_bilstm_bidirectional,
            batch_first=True,
        )
        self.audio_lstm = nn.LSTM(
            input_size=F,
            hidden_size=cfg.fusion_bilstm_hidden_audio,
            num_layers=cfg.fusion_bilstm_layers,
            dropout=cfg.fusion_bilstm_dropout if cfg.fusion_bilstm_layers > 1 else 0.0,
            bidirectional=cfg.fusion_bilstm_bidirectional,
            batch_first=True,
        )

        t_out_dim = 2 * cfg.fusion_bilstm_hidden_text if cfg.fusion_bilstm_bidirectional else cfg.fusion_bilstm_hidden_text
        a_out_dim = 2 * cfg.fusion_bilstm_hidden_audio if cfg.fusion_bilstm_bidirectional else cfg.fusion_bilstm_hidden_audio
        self.text_proj = nn.Sequential(nn.Linear(t_out_dim, F), nn.LayerNorm(F))
        self.audio_proj = nn.Sequential(nn.Linear(a_out_dim, F), nn.LayerNorm(F))

        self.blocks_t = nn.ModuleList([_CrossBlock(F, cfg.num_attention_head, cfg.dropout) for _ in range(cfg.fusion_blocks)])
        self.blocks_a = nn.ModuleList([_CrossBlock(F, cfg.num_attention_head, cfg.dropout) for _ in range(cfg.fusion_blocks)])


        self.use_gate = (cfg.fusion_merge == "gate")
        if self.use_gate:
            self.gate_t = nn.Sequential(nn.Linear(F, F), nn.GELU(), nn.Linear(F, F), nn.Sigmoid())
            self.gate_a = nn.Sequential(nn.Linear(F, F), nn.GELU(), nn.Linear(F, F), nn.Sigmoid())


        self.pool_attn = AttnPool(F, num_heads=cfg.fusion_pool_heads, dropout=cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    @staticmethod
    def _pack_lstm(x: torch.Tensor, kpm: torch.Tensor | None, lstm: nn.LSTM) -> torch.Tensor:

        if kpm is None:
            out, _ = lstm(x)
            return out

        lengths = (~kpm).sum(dim=1).to(torch.long).clamp(min=1)
        sorted_len, idx = torch.sort(lengths, descending=True)
        x_sorted = x.index_select(0, idx)

        packed = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_len.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = lstm(packed)
        out_sorted, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        inv_idx = torch.empty_like(idx)
        inv_idx[idx] = torch.arange(idx.size(0), device=idx.device)
        out = out_sorted.index_select(0, inv_idx)
        return out

    def forward(self, t, a, kpm_text=None, kpm_audio=None):

        t_lstm = self._pack_lstm(t, kpm_text, self.text_lstm)
        a_lstm = self._pack_lstm(a, kpm_audio, self.audio_lstm)


        t1 = self.text_proj(t_lstm)
        a1 = self.audio_proj(a_lstm)


        attn_cache = {"text": [], "audio": []}
        for bt, ba in zip(self.blocks_t, self.blocks_a):
            t1, w_t = bt(t1, a1, kpm_kv=kpm_audio)
            a1, w_a = ba(a1, t1, kpm_kv=kpm_text)
            attn_cache["text"].append(w_t)
            attn_cache["audio"].append(w_a)


        if self.use_gate:
            t1 = self.gate_t(t1) * t1
            a1 = self.gate_a(a1) * a1


        fusion_seq = torch.cat([t1, a1], dim=1)
        fusion_seq = self.drop(fusion_seq)


        pooled = None
        if self.cfg.fusion_head_output_type == "attn":
            if (kpm_text is not None) and (kpm_audio is not None):
                kpm_fusion = torch.cat([kpm_text, kpm_audio], dim=1)
            elif kpm_text is not None:
                kpm_fusion = torch.cat(
                    [kpm_text, torch.zeros(fusion_seq.size(0), a1.size(1), device=fusion_seq.device, dtype=torch.bool)],
                    dim=1
                )
            elif kpm_audio is not None:
                kpm_fusion = torch.cat(
                    [torch.zeros(fusion_seq.size(0), t1.size(1), device=fusion_seq.device, dtype=torch.bool), kpm_audio],
                    dim=1
                )
            else:
                kpm_fusion = None
            pooled = self.pool_attn(fusion_seq, kpm=kpm_fusion)

        return fusion_seq, pooled, attn_cache
