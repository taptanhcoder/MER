import torch.nn as nn
from transformers import AutoConfig, AutoModel

def build_phobert_encoder(model_name: str = "vinai/phobert-base") -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=False,
    )
    return AutoModel.from_pretrained(model_name, config=config)
