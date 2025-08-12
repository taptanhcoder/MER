import torch
import torch.nn as nn
from torch import hub

VGGISH_WEIGHTS = "https://github.com/namphuongtran9196/GitReleaseStorage/releases/download/torchvggish/vggish-10086976.pth"
PCA_PARAMS     = "https://github.com/namphuongtran9196/GitReleaseStorage/releases/download/torchvggish/vggish_pca_params-970ea276.pth"


class Postprocessor(nn.Module):
    """
    PCA (whitening) + clip + quantize như YouTube-8M.
    Dùng cho inference/compat; KHÔNG khuyến nghị khi training end-to-end.
    """
    def __init__(self, pca_params_path: str = None):
        super().__init__()
        if pca_params_path is None:
            params = hub.load_state_dict_from_url(PCA_PARAMS, map_location="cpu")
        else:
            params = torch.load(pca_params_path, map_location="cpu")
        self.register_buffer("_pca_matrix", torch.as_tensor(params["pca_eigen_vectors"]).float())
        self.register_buffer("_pca_means",  torch.as_tensor(params["pca_means"].reshape(-1, 1)).float())

    @torch.no_grad()
    def postprocess(self, embeddings_batch: torch.Tensor) -> torch.Tensor:
        pca_applied = (embeddings_batch.t() - self._pca_means)          
        pca_applied = torch.matmul(self._pca_matrix, pca_applied).t()     
        clipped = torch.clamp(pca_applied, -2.0, +2.0)
        quantized = torch.round((clipped + 2.0) * (255.0 / 4.0))           
        return quantized  

# ---- VGGish backbone ----
def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    """
    Input:  (B, 1, 96, 64)  log-mel patches
    Output: (B, 128) embeddings; nếu postprocess=True: PCA+quantized float
    """
    def __init__(self, features: nn.Module, postprocess: bool):
        super().__init__()
        self.postprocess = postprocess
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )
        self.pproc = Postprocessor() if postprocess else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.dim() == 4 and x.size(1) == 1 and x.size(2) == 96 and x.size(3) == 64, \
            f"VGGish expects (B,1,96,64), got {tuple(x.shape)}"
        x = self.features(x)             
        x = x.transpose(1, 3).transpose(1, 2).contiguous()  
        x = x.view(x.size(0), -1)           
        x = self.embeddings(x)             
        if self.postprocess:
            x = self.pproc.postprocess(x) 
        return x

def _vgg(postprocess=False) -> VGG:
    return VGG(make_layers(), postprocess)

def vggish(postprocess: bool = False,
           weights_path: str = None,
           pca_params_path: str = None,
           freeze_feature: bool = True) -> VGG:
    """
    Tạo VGGish model.
    - postprocess=False khi TRAIN (tránh mất gradient sau FC).
    - weights_path/pca_params_path: dùng local nếu có (offline).
    - freeze_feature=True: đóng băng conv, chỉ train FC.
    """
    model = _vgg(postprocess=postprocess)
    if weights_path is None:
        state_dict = hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True, map_location="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if postprocess:
        model.pproc = Postprocessor(pca_params_path)

    if freeze_feature:
        for p in model.features.parameters():
            p.requires_grad = False

    return model
