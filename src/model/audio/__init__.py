
from typing import Optional

from configs.base import Config

from .wav2vec2_xlsr import Wav2Vec2XLSR
from .fourier2vec import Fourier2Vec
from .phowhisper import PhoWhisperEncoder

from .dual_w2v2_vggish import build_w2v2_plus_vggish_encoder


def build_wav2vec2_xlsr_encoder(cfg: Config):
    enc = Wav2Vec2XLSR(cfg.audio_encoder_ckpt, trainable=bool(cfg.audio_unfreeze))
    # Đồng bộ meta dim
    if not hasattr(enc, "config"):
        class C: ...
        enc.config = C()
    enc.config.hidden_size = getattr(enc, "hidden_size", None) or 1024
    return enc


def build_fourier2vec_encoder(cfg: Config):
    return Fourier2Vec(cfg)


def build_phowhisper_encoder(cfg: Config):

    return PhoWhisperEncoder(
        model_name_or_path=cfg.audio_encoder_ckpt,
        sample_rate=int(getattr(cfg, "sample_rate", 16000)),
        max_audio_sec=cfg.max_audio_sec,
        # Override tuỳ chọn (None -> dùng mặc định của Whisper)
        n_fft=cfg.whisper_n_fft,
        hop_length=cfg.whisper_hop_length,
        win_length=cfg.whisper_win_length,
        nb_mels=cfg.whisper_nb_mels,
        trainable=bool(cfg.audio_unfreeze),
    )


def build_audio_encoder(cfg: Config):

    t = (cfg.audio_encoder_type or "").lower()
    if t == "wav2vec2_xlsr":
        return build_wav2vec2_xlsr_encoder(cfg)
    if t == "fourier2vec":
        return build_fourier2vec_encoder(cfg)
    if t == "phowhisper":
        return build_phowhisper_encoder(cfg)
    if t == "w2v2_vggish":
        # sử dụng tham số merge từ config (concat|sum)
        return build_w2v2_plus_vggish_encoder(cfg)
    raise ValueError(f"Unsupported audio encoder_type={cfg.audio_encoder_type!r}")
