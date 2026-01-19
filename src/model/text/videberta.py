import torch.nn as nn
from transformers import AutoConfig, AutoModel

def build_videberta_encoder(model_name: str) -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=False,
    )
    return AutoModel.from_pretrained(model_name, config=config)
