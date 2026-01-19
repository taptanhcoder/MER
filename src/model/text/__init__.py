import torch.nn as nn
from typing import Optional
from .phobert import build_phobert_encoder
from .videberta import build_videberta_encoder

def build_text_encoder(type: str = "phobert", model_name: Optional[str] = None) -> nn.Module:
    encoders = {
        "phobert":   build_phobert_encoder,
        "videberta": build_videberta_encoder,
    }
    assert type in encoders, f"Invalid text encoder type: {type}"
    if model_name is not None:
        return encoders[type](model_name)
    if type == "videberta":
        raise ValueError("Please provide `model_name` (ckpt) for ViDeBERTa via cfg.text_encoder_ckpt")
    return encoders[type]()
