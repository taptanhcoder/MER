import torch
import torch.nn as nn


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

class VGGishEncoder(nn.Module):

    def __init__(self, postprocess: bool = False, freeze_feature: bool = True):
        super().__init__()
        if not _HAS_VGGISH:
            raise RuntimeError(
                "VGGish modules not available. Ensure src/torchvggish is importable "
                "and its dependencies are installed."
            )
        self.model = build_vggish_backbone(postprocess=postprocess, freeze_feature=freeze_feature)
        self.config = type("Cfg", (), {"hidden_size": int(VGG_DIM)})

    @torch.no_grad()
    def get_feat_lengths(self, lengths: torch.Tensor) -> torch.Tensor:

        T = lengths.to(torch.long)
        win = int(round(VGG_WIN_SEC * VGG_SR))
        hop = int(round(VGG_HOP_SEC * VGG_SR))
        L = 1 + torch.div(torch.clamp(T - win, min=0), hop, rounding_mode="floor")
        return L.clamp(min=1)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        device = x.device
        B = x.size(0)
        per_lengths = []
        per_feats = []

        for b in range(B):
            xb = x[b].detach().cpu().numpy()
            ex = vggish_waveform_to_examples(xb, sample_rate=VGG_SR, return_tensor=True, device=device)  
            if ex.dim() == 3:
                ex = ex.unsqueeze(0)
            emb = self.model(ex)                
            per_feats.append(emb)
            per_lengths.append(emb.size(0))

        L_max = max(int(l) for l in per_lengths) if per_lengths else 1
        out = x.new_zeros((B, L_max, int(VGG_DIM)))
        for b, emb in enumerate(per_feats):
            Lb = emb.size(0)
            if Lb > 0:
                out[b, :Lb, :] = emb
        return out
