import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, Wav2Vec2Model
from configs.base import Config


def build_phobert_encoder(model_name: str = "vinai/phobert-base") -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=False,
    )
    return AutoModel.from_pretrained(model_name, config=config)

def build_text_encoder(type: str = "phobert", model_name: str = None) -> nn.Module:
    encoders = {
        "phobert": build_phobert_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    if model_name is not None:
        return encoders[type](model_name)
    return encoders[type]()


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


def build_wav2vec2_xlsr_encoder(cfg: Config) -> nn.Module:
    ckpt = getattr(cfg, "audio_encoder_ckpt", "facebook/wav2vec2-large-xlsr-53")
    return Wav2Vec2XLSR(ckpt=ckpt, trainable=cfg.audio_unfreeze)

def build_audio_encoder(cfg: Config) -> nn.Module:
    type_ = cfg.audio_encoder_type
    encoders = {
        "wav2vec2_xlsr": build_wav2vec2_xlsr_encoder,
    }
    assert type_ in encoders.keys(), f"Invalid audio encoder type: {type_}"
    return encoders[type_](cfg)
