# -*- coding: utf-8 -*-
# src/train_mer_videberta_w2v2vgg_bilstm.py

import os
import json
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, BatchEncoding

# Reuse training utilities
from train import Trainer, compute_class_weights_lenfreq
from configs.base import Config
from loading.dataloader import build_train_test_dataset, VNEMOSDataset
from model.networks import MER
from model.losses import CrossEntropyLoss
from training.callbacks import CheckpointsCallback
from training.optimizers import split_param_groups, build_optimizer

# ===== Quiet mode (ẩn bớt warning/log) =====
QUIET = os.getenv("QUIET", "0").strip() == "1"
if QUIET:
    warnings.filterwarnings("ignore")
    try:
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
    except Exception:
        pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ===== Early Stopping khớp chữ ký callbacks của Trainer.fit =====
class _StopTraining(Exception):
    pass

class EarlyStoppingCallback:
    """Dừng sớm khi metric không cải thiện sau N lần (patience)."""
    def __init__(self, monitor: str = "val_macro_f1", mode: str = "max",
                 patience: int = 5, min_delta: float = 0.0):
        self.monitor = monitor
        self.mode = mode.lower().strip()
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.num_bad = 0

    # Khớp chữ ký: (trainer, global_step, global_epoch, logs, isValPhase=False, logger=...)
    def __call__(self, trainer, global_step, global_epoch, logs, isValPhase=False, logger=None):
        if not isValPhase or logs is None or self.monitor not in logs:
            return
        val = float(logs[self.monitor])

        if self.best is None:
            self.best = val
            self.num_bad = 0
            return

        improved = (val > self.best + self.min_delta) if self.mode == "max" else (val < self.best - self.min_delta)
        if improved:
            self.best = val
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                if not QUIET:
                    print(f"[EarlyStopping] Patience reached ({self.patience}). Stop training.")
                raise _StopTraining()


def _project_root() -> Path:
    # src/ -> project root
    return Path(__file__).resolve().parents[1]


def main():
    PROJ = _project_root()
    ckpt_dir = PROJ / "checkpoints" / "videberta_w2v2vgg_bilstm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ENV overrides
    text_ckpt = os.getenv("TEXT_ENCODER_CKPT", (PROJ / "videberta-base").as_posix()).strip()
    if not Path(text_ckpt).exists() and "/" not in text_ckpt:
        # nếu không phải path local, dùng repo id mặc định
        text_ckpt = "Fsoft-AIC/videberta-base"

    w2v2_ckpt = os.getenv("W2V2_CKPT", "facebook/wav2vec2-large-xlsr-53").strip()
    merge_mode = os.getenv("W2V2_VGGISH_MERGE", "concat").strip().lower()
    if merge_mode not in {"concat", "sum"}:
        merge_mode = "concat"

    cfg = Config(
        # run meta
        name="MER_ViDeBERTa_W2V2VGG_BiLSTM",
        checkpoint_dir=str(ckpt_dir),
        num_epochs=50,
        batch_size=8,
        num_workers=2,

        # tối ưu
        learning_rate=2e-4,            # LR cho head; encoder dùng LR nhỏ qua split_param_groups()
        optimizer_type="AdamW",
        scheduler_type="cosine_warmup",
        warmup_ratio=0.05,
        scheduler_step_unit="step",

        # loss
        loss_type="CrossEntropyLoss",
        label_smoothing=0.0,

        # data (tương đối theo project-root)
        data_root=(PROJ / "output").as_posix(),
        jsonl_dir="",
        sample_rate=16000,
        max_audio_sec=None,
        text_max_length=96,

        # sampling
        use_length_bucket=True,
        length_bucket_size=64,
        bucketing_text_alpha=0.03,
        use_weighted_sampler=True,
        lenfreq_alpha=0.5,

        # MODEL
        model_type="MemoCMT",

        # Text encoder → ViDeBERTa
        text_encoder_type="videberta",        # {"phobert","videberta"}
        text_encoder_ckpt=text_ckpt,
        text_encoder_dim=768,
        text_unfreeze=False,

        # Audio encoder → W2V2 + VGGish (dual)
        audio_encoder_type="w2v2_vggish",     # {"wav2vec2_xlsr","fourier2vec","phowhisper","w2v2_vggish"}
        audio_encoder_ckpt=w2v2_ckpt,         # W2V2 backbone; VGGish dùng internal weights
        audio_encoder_dim=1024,               # sẽ overwrite runtime nếu cần
        audio_unfreeze=False,
        w2v2_vggish_merge=merge_mode,         # "concat" | "sum"

        # Fusion = Bi-LSTM Attention
        fusion_type="bilstm_attn",
        fusion_dim=768,
        fusion_head_output_type="max",        # 'cls' | 'mean' | 'max' | 'min' | 'attn'
        linear_layer_output=[256, 128],
        dropout=0.10,

        # Hyper cho Fusion-B (Bi-LSTM Attention)
        fusion_bilstm_hidden_text=384,
        fusion_bilstm_hidden_audio=384,
        fusion_bilstm_layers=1,
        fusion_bilstm_dropout=0.1,
        fusion_bilstm_bidirectional=True,
        fusion_blocks=1,
        fusion_merge="concat",                # "concat" | "gate"
        fusion_pool_heads=1,                  # dùng nếu fusion_head_output_type == "attn"

        # Tricks
        use_amp=True,
        max_grad_norm=1.0,

        save_best_val=True,
        max_to_keep=2,
    )

    # === Data ===
    train_loader, eval_loader = build_train_test_dataset(cfg)
    _train_ds = VNEMOSDataset(cfg, "train")
    label2id = _train_ds.label2id
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    cfg.num_classes = len(id2label)

    # === Model ===
    device = "cuda" if (torch.cuda.is_available() and os.getenv("USE_CUDA", "1") != "0") else "cpu"
    net = MER(cfg, device=device).to(device)

    # === Class weights (len-freq) ===
    class_weights = compute_class_weights_lenfreq(_train_ds, alpha=float(getattr(cfg, "lenfreq_alpha", 0.5))).to(device)
    if not QUIET:
        print("Class weights (len-freq):", class_weights.tolist())
    criterion = CrossEntropyLoss(cfg, weight=class_weights)

    trainer = Trainer(cfg, net, criterion, log_dir="logs")

    # === Optimizer & Scheduler ===
    enc_lr, head_lr = 5e-6, float(cfg.learning_rate)     # encoder LR nhỏ; head LR lớn hơn
    param_groups = split_param_groups(trainer, lr_enc=enc_lr, lr_head=head_lr, weight_decay=0.05)
    optimizer = build_optimizer("adamw", param_groups, lr=head_lr, weight_decay=0.05)

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    trainer.compile(
        optimizer=optimizer,
        scheduler=scheduler,
        lr=head_lr,
        param_groups=None,
        scheduler_step_unit=cfg.scheduler_step_unit
    )

    # === Callbacks: checkpoint + early stop (tùy chọn) ===
    ckpt_cb = CheckpointsCallback(
        cfg.checkpoint_dir,
        save_freq=200,
        max_to_keep=cfg.max_to_keep,
        save_best_val=True,
        monitor="val_macro_f1",
        mode="max",
    )
    early_cb = EarlyStoppingCallback(
        monitor="val_macro_f1",
        mode="max",
        patience=int(os.getenv("EARLY_PATIENCE", "5")),
        min_delta=0.0,
    )
    callbacks = [ckpt_cb, early_cb]

    # === Train ===
    try:
        trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=eval_loader, callbacks=callbacks)
    except _StopTraining:
        if not QUIET:
            print("Training stopped early by EarlyStopping.")

    best_path = getattr(ckpt_cb, "best_path", "")
    if not QUIET:
        print("Best checkpoint:", best_path)

    # (optional) load best weights
    if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
        if not QUIET:
            print("Loading best weights:", best_path)
        state = torch.load(best_path, map_location=trainer.device)
        trainer.network.load_state_dict(state)

    # === Evaluate & Report ===
    def collect_preds(_trainer, loader):
        all_preds, all_labels = [], []
        _trainer.network.eval()
        with torch.no_grad():
            for batch in loader:
                tok, audio, labels, meta = _trainer._unpack_batch(batch)
                audio  = audio.to(_trainer.device)
                labels = labels.to(_trainer.device)
                # move text to device
                if isinstance(tok, BatchEncoding):
                    tok = tok.to(_trainer.device); tok = {k: v for k, v in tok.items()}
                elif isinstance(tok, dict):
                    tok = {k: v.to(_trainer.device) for k, v in tok.items()}
                else:
                    tok = tok.to(_trainer.device)

                out = _trainer.network(tok, audio, meta=meta)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_preds), torch.cat(all_labels)

    if eval_loader is not None:
        preds, labels = collect_preds(trainer, eval_loader)
        num_classes = cfg.num_classes

        cm = torch.zeros((num_classes, num_classes), dtype=torch.int32)
        for t, p in zip(labels, preds):
            cm[t.long(), p.long()] += 1

        per_class_acc = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()).numpy()
        overall_acc = (preds == labels).float().mean().item()

        print("=== Evaluation ===")
        print("Overall Acc: %.4f" % overall_acc)
        print("Per-class Acc:")
        id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
        for i, acc in enumerate(per_class_acc):
            name = id2label[i] if i < len(id2label) else str(i)
            print(f"  {i} ({name}): {acc:.4f}")
        print("Confusion Matrix:\n", cm.numpy())

        try:
            from sklearn.metrics import classification_report
            target_names = [id2label[i] for i in range(num_classes)]
            print("\nClassification Report:")
            print(classification_report(labels.numpy(), preds.numpy(), target_names=target_names, digits=4))
        except Exception as e:
            print("sklearn not available:", e)
    else:
        print("No eval_loader available.")

    # === Export HDF5 (.h5) vào checkpoints/videberta_w2v2vgg_bilstm ===
    try:
        import h5py
        h5_name = "model_weights_videberta_w2v2vgg_bilstm.h5"
        h5_path = ckpt_dir / h5_name

        state = trainer.network.state_dict()
        torch_ver = str(getattr(torch, "__version__", "unknown"))
        id2label_json = json.dumps([k for k, v in sorted(label2id.items(), key=lambda x: x[1])], ensure_ascii=False)

        with h5py.File(h5_path, "w") as f:
            f.attrs["model_name"] = "MER"
            f.attrs["num_classes"] = int(cfg.num_classes)
            f.attrs["id2label"] = id2label_json
            f.attrs["torch_version"] = torch_ver

            cfg_group = f.create_group("config")
            for k, v in vars(cfg).items():
                try:
                    cfg_group.attrs[k] = json.dumps(v, ensure_ascii=False)
                except TypeError:
                    cfg_group.attrs[k] = str(v)

            weights_group = f.create_group("state_dict")
            for name, tensor in state.items():
                arr = tensor.detach().cpu().numpy()
                dset = weights_group.create_dataset(name, data=arr)
                dset.attrs["shape"] = arr.shape
                dset.attrs["dtype"] = str(arr.dtype)

        print(f" Saved HDF5 weights to: {h5_path.resolve()}")
    except Exception as e:
        print("[WARN] Export .h5 failed:", e)


if __name__ == "__main__":
    main()
