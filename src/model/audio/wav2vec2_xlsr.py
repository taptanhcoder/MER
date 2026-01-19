import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

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
