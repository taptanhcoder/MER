from torch import optim

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



def split_param_groups(model,
                       lr_enc: float = 1e-5,
                       lr_head: float = 1e-4,
                       weight_decay: float = 0.01):

    module = getattr(model, "network", model)
    enc_prefixes = ("text_encoder", "audio_encoder")
    enc_ids = set()
    enc, head = [], []

    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(enc_prefixes):
            enc.append(p)
            enc_ids.add(id(p))
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if id(p) not in enc_ids:
            head.append(p)

    groups = []
    if enc:  groups.append({"params": enc,  "lr": lr_enc,  "weight_decay": weight_decay})
    if head: groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
    return groups


def build_optimizer(name: str,
                    params_or_groups,
                    lr: float = 1e-4,
                    weight_decay: float = 0.01,
                    betas=(0.9, 0.999),
                    eps: float = 1e-8,
                    momentum: float = 0.9):
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
