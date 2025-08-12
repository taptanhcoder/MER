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

def build_text_encoder(type: str = "phobert") -> nn.Module:

    encoders = {
        "phobert": build_phobert_encoder,  
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()


class Wav2Vec2XLSR(nn.Module):

    def __init__(self, ckpt: str = "facebook/wav2vec2-large-xlsr-53", trainable: bool = False):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(ckpt)
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.model(x)                 
        return out.last_hidden_state       

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
