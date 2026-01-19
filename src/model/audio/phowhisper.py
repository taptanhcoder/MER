from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel


@dataclass
class _WhisperSpec:
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    win_length: Optional[int] = None
    nb_mels: Optional[int] = None

    def effective(self, fe: WhisperFeatureExtractor):
        n_fft = self.n_fft if self.n_fft is not None else getattr(fe, "n_fft", 400)
        hop = self.hop_length if self.hop_length is not None else getattr(fe, "hop_length", 160)  
        win = self.win_length if self.win_length is not None else getattr(fe, "win_length", 400)
        n_mels = self.nb_mels if self.nb_mels is not None else getattr(fe, "nb_mels", 80)
        return n_fft, hop, win, n_mels


class PhoWhisperEncoder(nn.Module):


    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-small",
        sample_rate: int = 16000,
        max_audio_sec: Optional[float] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        nb_mels: Optional[int] = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.max_audio_sec = max_audio_sec
        self.spec = _WhisperSpec(n_fft=n_fft, hop_length=hop_length, win_length=win_length, nb_mels=nb_mels)

        # Load FE + model
        self.fe = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
        # Ghi đè tham số FE nếu user truyền vào
        if n_fft is not None:
            self.fe.n_fft = int(n_fft)
        if hop_length is not None:
            self.fe.hop_length = int(hop_length)
        if win_length is not None:
            self.fe.win_length = int(win_length)
        if nb_mels is not None:
            self.fe.nb_mels = int(nb_mels)

        self.model = WhisperModel.from_pretrained(model_name_or_path)

        # Expose hidden_size
        if not hasattr(self, "config"):
            class C: ...
            self.config = C()
        self.config.hidden_size = int(self.model.config.d_model)

        # Freeze nếu không train
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

        # Mặc định giới hạn 30s giống Whisper
        self.default_max_audio_samples = 30 * self.sample_rate

    @torch.no_grad()
    def _truncate_waveform(self, x: torch.Tensor) -> torch.Tensor:

        max_samples_cfg = None
        if self.max_audio_sec is not None and self.max_audio_sec > 0:
            max_samples_cfg = int(round(self.max_audio_sec * self.sample_rate))

        max_samples = max_samples_cfg or self.default_max_audio_samples
        if x.size(-1) > max_samples:
            x = x[..., :max_samples]
        return x

    @torch.no_grad()
    def get_feat_lengths(self, lengths_in_samples: torch.Tensor) -> torch.Tensor:

        device = lengths_in_samples.device
        _, hop, win, _ = self.spec.effective(self.fe)

        # Áp trần theo max_audio_sec hoặc 30s mặc định
        if self.max_audio_sec is not None and self.max_audio_sec > 0:
            cap = int(round(self.max_audio_sec * self.sample_rate))
            lengths_in_samples = torch.minimum(lengths_in_samples, torch.tensor(cap, device=device))
        else:
            cap = self.default_max_audio_samples
            lengths_in_samples = torch.minimum(lengths_in_samples, torch.tensor(cap, device=device))

        L_minus = (lengths_in_samples - win).clamp(min=0)
        stft_T = torch.floor_divide(L_minus, hop) + 1
        enc_T = torch.clamp(torch.floor_divide(stft_T, 2), min=1)  # downsample ~2x
        return enc_T

    def _to_numpy_list(
        self,
        x_cpu: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        max_samples: Optional[int] = None,
    ) -> List[np.ndarray]:

        B, T = x_cpu.shape
        lengths = None
        if attn_mask is not None:
            # attn_mask: 1=real, 0=pad
            lengths = attn_mask.detach().sum(dim=1).cpu().tolist()

        waves: List[np.ndarray] = []
        for i in range(B):
            if lengths is not None:
                L = int(lengths[i])
                L = max(0, min(L, T))
                w = x_cpu[i, :L]
            else:
                w = x_cpu[i]

            if max_samples is not None and w.numel() > max_samples:
                w = w[:max_samples]

            waves.append(w.contiguous().numpy().astype(np.float32, copy=False))
        return waves

    def forward(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        was_training = self.training
        self.eval()
        try:
            if waveform.dim() != 2:
                raise ValueError(f"PhoWhisperEncoder expects waveform shape (B,T), got {tuple(waveform.shape)}")

            # Chuẩn bị device và max samples
            device = waveform.device
            max_samples_cfg = None
            if self.max_audio_sec is not None and self.max_audio_sec > 0:
                max_samples_cfg = int(round(self.max_audio_sec * self.sample_rate))
            max_samples = max_samples_cfg or self.default_max_audio_samples

            # FE yêu cầu CPU numpy → đưa về CPU trước khi extract
            x_cpu = waveform.detach().cpu()
            attn_cpu = attention_mask.detach().cpu() if attention_mask is not None else None

            # Tạo list numpy, có cắt theo attention_mask & max_samples
            waves = self._to_numpy_list(x_cpu, attn_mask=attn_cpu, max_samples=max_samples)

            # Gọi FE trên CPU
            fe_kwargs = dict(sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = self.fe(waves, **fe_kwargs)  # BatchFeature
            feats = inputs["input_features"]  # (B, n_mels, stft_T) theo định dạng của Whisper

            # Đưa features về device gốc và pass vào encoder
            feats = feats.to(device)
            enc = self.model.encoder(input_features=feats)
            h = enc.last_hidden_state  # (B, L', D)
            return h
        finally:
            self.train(was_training)
