from torch import optim
from typing import Iterable, List, Dict, Any

# Giữ các hàm tiện dụng cũ
def sgd(params, learning_rate=0.01, momentum=0.9):
    return optim.SGD(params, lr=learning_rate, momentum=momentum)

def adam(params, learning_rate=0.01):
    return optim.Adam(params, lr=learning_rate)

def rmsprop(params, learning_rate=0.01):
    return optim.RMSprop(params, lr=learning_rate)

def adagrad(params, learning_rate=0.01):
    return optim.Adagrad(params, lr=learning_rate)

def adamw(params, learning_rate=0.01, weight_decay=0.01):
    return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def _param_is_encoder(name: str, enc_prefixes=("text_encoder", "audio_encoder")) -> bool:
    return name.startswith(enc_prefixes)


def _no_decay_name(name: str, no_decay=("bias", "LayerNorm.weight")) -> bool:
    return any(nd in name for nd in no_decay)


def split_param_groups(model,
                       lr_enc: float = 1e-5,
                       lr_head: float = 1e-4,
                       weight_decay: float = 0.01,
                       no_decay_keys=("bias", "LayerNorm.weight")) -> List[Dict[str, Any]]:
    """
    Tách tham số thành 4 nhóm:
      - encoder_decay,    encoder_no_decay
      - head_decay,       head_no_decay
    Mỗi nhóm có LR riêng (enc/head) và decay khác nhau (0 cho no_decay).
    Giữ nguyên hành vi: nếu nhóm trống sẽ không trả về nhóm đó.
    """
    module = getattr(model, "network", model)

    enc_decay, enc_no_decay, head_decay, head_no_decay = [], [], [], []

    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if _param_is_encoder(n):
            if _no_decay_name(n, no_decay=no_decay_keys):
                enc_no_decay.append(p)
            else:
                enc_decay.append(p)
        else:
            if _no_decay_name(n, no_decay=no_decay_keys):
                head_no_decay.append(p)
            else:
                head_decay.append(p)

    groups: List[Dict[str, Any]] = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": lr_enc, "weight_decay": weight_decay})
    if enc_no_decay:
        groups.append({"params": enc_no_decay, "lr": lr_enc, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})
    return groups


def build_optimizer(name: str,
                    params_or_groups,
                    lr: float = 1e-4,
                    weight_decay: float = 0.01,
                    betas=(0.9, 0.999),
                    eps: float = 1e-8,
                    momentum: float = 0.9):
    """
    Hỗ trợ cả params list và param groups (từ split_param_groups()).
    """
    name = name.lower()
    if name == "adamw":
        return optim.AdamW(params_or_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "adam":
        return optim.Adam(params_or_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params_or_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params_or_groups, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "adagrad":
        return optim.Adagrad(params_or_groups, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")
