import math
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, Wav2Vec2Model
from configs.base import Config


try:
    from torchvggish.torchvggish import vggish as build_vggish_backbone
    from torchvggish.vggish_input import waveform_to_examples as vggish_waveform_to_examples
    from torchvggish.vggish_params import (
        EXAMPLE_WINDOW_SECONDS as VGG_WIN_SEC,
        EXAMPLE_HOP_SECONDS as VGG_HOP_SEC,
        SAMPLE_RATE as VGG_SR,
        EMBEDDING_SIZE as VGG_DIM,
    )
    _HAS_VGGISH = True
except Exception:

    _HAS_VGGISH = False


# ---------- TEXT ENCODERS ----------

def build_phobert_encoder(model_name: str = "vinai/phobert-base") -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=False,
    )
    return AutoModel.from_pretrained(model_name, config=config)

def build_videberta_encoder(model_name: str) -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=False,
    )
    return AutoModel.from_pretrained(model_name, config=config)

def build_text_encoder(type: str = "phobert", model_name: str = None) -> nn.Module:
    encoders = {
        "phobert":   build_phobert_encoder,
        "videberta": build_videberta_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    if model_name is not None:
        return encoders[type](model_name)
    if type == "videberta":
        raise ValueError("Please provide `model_name` (ckpt) for ViDeBERTa via cfg.text_encoder_ckpt")
    return encoders[type]()


# ---------- AUDIO ENCODERS ----------

class Wav2Vec2XLSR(nn.Module):
    def __init__(self, ckpt: str = "facebook/wav2vec2-large-xlsr-53", trainable: bool = False):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(ckpt)
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x, attention_mask=None):
        out = self.model(x, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state

    @torch.no_grad()
    def get_feat_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = self.model._get_feat_extract_output_lengths(lengths.detach().cpu())
        out = torch.as_tensor(out, dtype=torch.long, device=lengths.device)
        return out.clamp(min=1)


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

        # STFT params
        self.sr = sr
        self.win_length = int(round(sr * (win_ms / 1000.0)))
        self.hop_length = int(round(sr * (hop_ms / 1000.0)))
        self.n_fft = 1 << (self.win_length - 1).bit_length()
        self.register_buffer("hann_window", torch.hann_window(self.win_length), persistent=False)

        # Mel filterbank
        mel_fb = self._build_mel_filter(self.n_fft, sr, n_mels, fmin, fmax)
        self.register_buffer("mel_filter", mel_fb, persistent=True)

        # Projector
        self.proj = nn.Linear(n_mels, d_model)

        # Optional temporal patching
        if self.patch_len > 1:
            self.time_pool = nn.AvgPool1d(kernel_size=self.patch_len, stride=self.patch_hop, ceil_mode=False)
        else:
            self.time_pool = None

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=getattr(cfg, "dropout", 0.1),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # expose hidden_size
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
        """
        x: (B, T)
        return: log-mel (B, L_stft, n_mels)
        """
        # Ensure window device
        if self.hann_window.device != x.device:
            self.hann_window = self.hann_window.to(x.device)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            return_complex=True,
            center=True,               # IMPORTANT: center=True
            pad_mode="constant",
        )
        spec = stft.abs()                                    # (B, freq, frames)
        mel_filter = self.mel_filter.to(spec.device, spec.dtype)
        mel = torch.matmul(spec.transpose(1, 2), mel_filter) # (B, frames, n_mels)
        logmel = torch.log(mel + self.log_offset)
        return logmel

    @torch.no_grad()
    def get_feat_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Hiệu chỉnh đúng cho STFT(center=True).
        Với center=True, PyTorch sẽ pad mỗi phía ~ win_length//2.
        frames_stft = 1 + floor( max(T + 2*pad - win, 0) / hop ),
        pad = win//2 (chuẩn PyTorch).
        Sau đó áp dụng patching (nếu có).
        """
        T = lengths.to(torch.long)
        win = self.win_length
        hop = self.hop_length
        pad = win // 2

        # frames sau STFT (center=True)
        # đảm bảo không âm và clamp ≥1
        L = 1 + torch.div(torch.clamp(T + 2 * pad - win, min=0), hop, rounding_mode="floor")
        L = L.clamp(min=1)

        # patching nếu có
        if self.time_pool is not None:
            K = self.patch_len
            S = self.patch_hop
            L = 1 + torch.div(torch.clamp(L - K, min=0), S, rounding_mode="floor")
            L = L.clamp(min=1)
        return L

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        x: (B, T)
        """
        logmel = self._stft_logmel(x)          # (B, L_stft, n_mels)
        h = self.proj(logmel)                  # (B, L_stft, d_model)

        if self.time_pool is not None:
            h = h.transpose(1, 2)              # (B, d_model, L)
            h = self.time_pool(h)              # (B, d_model, L')
            h = h.transpose(1, 2)              # (B, L', d_model)
        out = self.encoder(h)                  # (B, L', d_model)
        return out


def build_wav2vec2_xlsr_encoder(cfg: Config) -> nn.Module:
    ckpt = getattr(cfg, "audio_encoder_ckpt", "facebook/wav2vec2-large-xlsr-53")
    return Wav2Vec2XLSR(ckpt=ckpt, trainable=cfg.audio_unfreeze)

def build_fourier2vec_encoder(cfg: Config) -> nn.Module:
    enc = Fourier2Vec(cfg)
    if not getattr(cfg, "audio_unfreeze", False):
        for p in enc.parameters():
            p.requires_grad = False
    return enc


# ---------- (NEW) VGGish Adapter & DualAudioEncoder ----------

class VGGishEncoder(nn.Module):
    """
    Adapter để VGGish tuân theo contract:
      - forward(B,T) -> (B, L_vgg, 128)
      - get_feat_lengths(T) -> L_vgg
      - config.hidden_size = 128
    """
    def __init__(self, postprocess: bool = False, freeze_feature: bool = True):
        super().__init__()
        if not _HAS_VGGISH:
            raise RuntimeError(
                "VGGish modules not available. Ensure src/torchvggish is importable "
                "and its dependencies are installed."
            )
        self.model = build_vggish_backbone(postprocess=postprocess, freeze_feature=freeze_feature)
        # expose hidden size like HF models
        self.config = type("Cfg", (), {"hidden_size": int(VGG_DIM)})

    @torch.no_grad()
    def get_feat_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Phù hợp với framing ở vggish_input.waveform_to_examples (0.96s window, hop 0.96s, min 1 frame).
        L = 1 + floor( max(T_samples - win_samples, 0) / hop_samples ), clamp >= 1
        """
        T = lengths.to(torch.long)
        win = int(round(VGG_WIN_SEC * VGG_SR))
        hop = int(round(VGG_HOP_SEC * VGG_SR))
        L = 1 + torch.div(torch.clamp(T - win, min=0), hop, rounding_mode="floor")
        return L.clamp(min=1)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        x: (B, T) waveform in [-1, 1]
        Trả về: (B, L_vgg, 128)
        """
        device = x.device
        B = x.size(0)
        per_lengths = []
        per_feats = []

        # chạy từng mẫu (VGGish util hiện thuần CPU/Numpy)
        for b in range(B):
            xb = x[b].detach().cpu().numpy()
            ex = vggish_waveform_to_examples(xb, sample_rate=VGG_SR, return_tensor=True, device=device)  # (N,1,96,64)
            # N có thể = 0 nếu input quá ngắn; vggish_input đã pad để có >=1 frame, nhưng vẫn phòng ngừa
            if ex.dim() == 3:  # thiếu batch dim?
                ex = ex.unsqueeze(0)
            emb = self.model(ex)                 # (N, 128)
            per_feats.append(emb)
            per_lengths.append(emb.size(0))

        L_max = max(int(l) for l in per_lengths) if per_lengths else 1
        out = x.new_zeros((B, L_max, int(VGG_DIM)))  # zeros trên cùng device/dtype

        for b, emb in enumerate(per_feats):
            Lb = emb.size(0)
            if Lb > 0:
                out[b, :Lb, :] = emb

        return out


class DualAudioEncoder(nn.Module):
    """
    Kết hợp W2V2 (L_ref cao) + VGGish (thưa) theo thời gian.
    Mặc định: nội suy VGGish theo trục thời gian -> concat kênh với W2V2.

    Trả về: (B, L_ref, D_total), get_feat_lengths = của W2V2.
    """
    def __init__(self, w2v2: Wav2Vec2XLSR, vggish: VGGishEncoder, mode: str = "concat"):
        super().__init__()
        assert mode in ("concat", "sum"), "mode must be 'concat' or 'sum'"
        self.w2v2 = w2v2
        self.vgg = vggish
        self.mode = mode

        if mode == "concat":
            hidden = self.w2v2.model.config.hidden_size + self.vgg.config.hidden_size
        else:
            hidden = max(self.w2v2.model.config.hidden_size, self.vgg.config.hidden_size)
        self.config = type("Cfg", (), {"hidden_size": int(hidden)})

        # Nếu muốn sum nhưng 2 chiều khác nhau, chiếu về cùng chiều
        if mode == "sum" and self.w2v2.model.config.hidden_size != self.vgg.config.hidden_size:
            d_w = int(self.w2v2.model.config.hidden_size)
            d_v = int(self.vgg.config.hidden_size)
            d_out = max(d_w, d_v)
            self.proj_w = nn.Linear(d_w, d_out, bias=False)
            self.proj_v = nn.Linear(d_v, d_out, bias=False)
        else:
            self.proj_w = None
            self.proj_v = None

    @torch.no_grad()
    def get_feat_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.w2v2.get_feat_lengths(lengths)

    def _time_interpolate(self, seq: torch.Tensor, L_ref: int) -> torch.Tensor:
        """
        seq: (B, L, D) -> nội suy time về L_ref
        """
        if seq.size(1) == L_ref:
            return seq
        # (B, D, L) để dùng F.interpolate(mode="linear")
        x = seq.transpose(1, 2)  # (B, D, L)
        x = nn.functional.interpolate(x, size=L_ref, mode="linear", align_corners=False)
        return x.transpose(1, 2)  # (B, L_ref, D)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        x: (B, T)
        """
        # 1) Đặc trưng W2V2 (L_ref cao)
        w = self.w2v2(x, attention_mask=attention_mask)   # (B, Lw, Dw)
        B, L_ref, Dw = w.size()

        # 2) Đặc trưng VGGish (thưa)
        v = self.vgg(x)                                    # (B, Lv, 128)

        # 3) Nội suy VGGish theo trục thời gian về L_ref
        v_interp = self._time_interpolate(v, L_ref)        # (B, L_ref, 128)

        # 4) Trộn
        if self.mode == "concat":
            out = torch.cat([w, v_interp], dim=-1)       
        else:
            a = self.proj_w(w) if self.proj_w is not None else w
            b = self.proj_v(v_interp) if self.proj_v is not None else v_interp
            out = a + b                                    # (B, L_ref, D_out)

        return out


def build_w2v2_plus_vggish_encoder(cfg: Config) -> nn.Module:

    if not _HAS_VGGISH:
        raise RuntimeError(
            "VGGish not available. Please ensure src/torchvggish is present and importable."
        )
    w2v2 = build_wav2vec2_xlsr_encoder(cfg)
    vgg = VGGishEncoder(postprocess=False, freeze_feature=True)
    enc = DualAudioEncoder(w2v2, vgg, mode=getattr(cfg, "w2v2_vggish_merge", "concat"))
    # freeze theo cfg.audio_unfreeze
    if not getattr(cfg, "audio_unfreeze", False):
        for p in enc.parameters():
            p.requires_grad = False
    return enc


def build_audio_encoder(cfg: Config) -> nn.Module:
    type_ = cfg.audio_encoder_type
    encoders = {
        "wav2vec2_xlsr": build_wav2vec2_xlsr_encoder,
        "fourier2vec":   build_fourier2vec_encoder,
        # (NEW)
        "w2v2_vggish":   build_w2v2_plus_vggish_encoder,
    }
    assert type_ in encoders.keys(), f"Invalid audio encoder type: {type_}"
    return encoders[type_](cfg)
