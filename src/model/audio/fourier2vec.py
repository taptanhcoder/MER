import torch
import torch.nn as nn
from configs.base import Config

class Fourier2Vec(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        sr = int(getattr(cfg, "sample_rate", 16000))
        n_mels = int(getattr(cfg, "fourier_n_mels", 64))
        fmin = float(getattr(cfg, "fourier_fmin", 125.0))
        fmax = float(getattr(cfg, "fourier_fmax", 7500.0))
        win_ms = float(getattr(cfg, "fourier_win_ms", 25.0))
        hop_ms = float(getattr(cfg, "fourier_hop_ms", 10.0))
        d_model = int(getattr(cfg, "fourier_hidden_size", 256))
        nhead = int(getattr(cfg, "fourier_num_heads", 4))
        nlayers = int(getattr(cfg, "fourier_num_layers", 4))
        self.patch_len = int(getattr(cfg, "fourier_patch_len", 1))
        self.patch_hop = int(getattr(cfg, "fourier_patch_hop", 1))
        self.log_offset = 1e-2

        self.sr = sr
        self.win_length = int(round(sr * (win_ms / 1000.0)))
        self.hop_length = int(round(sr * (hop_ms / 1000.0)))
        self.n_fft = 1 << (self.win_length - 1).bit_length()
        self.register_buffer("hann_window", torch.hann_window(self.win_length), persistent=False)

        mel_fb = self._build_mel_filter(self.n_fft, sr, n_mels, fmin, fmax)
        self.register_buffer("mel_filter", mel_fb, persistent=True)

        self.proj = nn.Linear(n_mels, d_model)

        if self.patch_len > 1:
            self.time_pool = nn.AvgPool1d(kernel_size=self.patch_len, stride=self.patch_hop, ceil_mode=False)
        else:
            self.time_pool = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=getattr(cfg, "dropout", 0.1),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.config = type("Cfg", (), {"hidden_size": d_model})

    @staticmethod
    def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
        return 1127.0 * torch.log1p(freq_hz / 700.0)

    @staticmethod
    def _build_mel_filter(n_fft: int, sr: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
        n_freq = n_fft // 2 + 1
        freqs = torch.linspace(0.0, sr / 2.0, n_freq)
        mel = Fourier2Vec._hz_to_mel(freqs)
        m_min = Fourier2Vec._hz_to_mel(torch.tensor(fmin))
        m_max = Fourier2Vec._hz_to_mel(torch.tensor(fmax))
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        fb = torch.zeros(n_freq, n_mels)
        for m in range(n_mels):
            m_l, m_c, m_r = m_pts[m], m_pts[m + 1], m_pts[m + 2]
            left = (mel - m_l) / (m_c - m_l)
            right = (m_r - mel) / (m_r - m_c)
            fb[:, m] = torch.clamp(torch.minimum(left, right), min=0.0)
        fb[0, :] = 0.0
        return fb

    def _stft_logmel(self, x: torch.Tensor) -> torch.Tensor:
        if self.hann_window.device != x.device:
            self.hann_window = self.hann_window.to(x.device)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            return_complex=True,
            center=True,           
            pad_mode="constant",
        )
        spec = stft.abs()                                  
        mel_filter = self.mel_filter.to(spec.device, spec.dtype)
        mel = torch.matmul(spec.transpose(1, 2), mel_filter) 
        logmel = torch.log(mel + self.log_offset)
        return logmel

    @torch.no_grad()
    def get_feat_lengths(self, lengths: torch.Tensor) -> torch.Tensor:

        T = lengths.to(torch.long)
        win = self.win_length
        hop = self.hop_length
        pad = win // 2

        L = 1 + torch.div(torch.clamp(T + 2 * pad - win, min=0), hop, rounding_mode="floor")
        L = L.clamp(min=1)

        if self.time_pool is not None:
            K = self.patch_len
            S = self.patch_hop
            L = 1 + torch.div(torch.clamp(L - K, min=0), S, rounding_mode="floor")
            L = L.clamp(min=1)
        return L

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        logmel = self._stft_logmel(x)         
        h = self.proj(logmel)                 

        if self.time_pool is not None:
            h = h.transpose(1, 2)             
            h = self.time_pool(h)              
            h = h.transpose(1, 2)              
        out = self.encoder(h)                  
        return out
