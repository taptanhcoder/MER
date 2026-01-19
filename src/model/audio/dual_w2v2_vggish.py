import torch
import torch.nn as nn
from configs.base import Config
from .wav2vec2_xlsr import Wav2Vec2XLSR
from .vggish_adapter import VGGishEncoder

class DualAudioEncoder(nn.Module):

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
        if seq.size(1) == L_ref:
            return seq
        x = seq.transpose(1, 2)  
        x = nn.functional.interpolate(x, size=L_ref, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        w = self.w2v2(x, attention_mask=attention_mask)  
        B, L_ref, _ = w.size()
        v = self.vgg(x)                                 
        v_interp = self._time_interpolate(v, L_ref)       

        if self.mode == "concat":
            out = torch.cat([w, v_interp], dim=-1)         
        else:
            a = self.proj_w(w) if self.proj_w is not None else w
            b = self.proj_v(v_interp) if self.proj_v is not None else v_interp
            out = a + b                                   
        return out

def build_w2v2_plus_vggish_encoder(cfg: Config) -> nn.Module:
    w2v2 = Wav2Vec2XLSR(ckpt=getattr(cfg, "audio_encoder_ckpt", "facebook/wav2vec2-large-xlsr-53"),
                        trainable=getattr(cfg, "audio_unfreeze", False))
    vgg = VGGishEncoder(postprocess=False, freeze_feature=True)
    enc = DualAudioEncoder(w2v2, vgg, mode=getattr(cfg, "w2v2_vggish_merge", "concat"))
    if not getattr(cfg, "audio_unfreeze", False):
        for p in enc.parameters():
            p.requires_grad = False
    return enc
