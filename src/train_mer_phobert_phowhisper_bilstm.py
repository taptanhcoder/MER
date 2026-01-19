# -*- coding: utf-8 -*-
# src/train_mer_videberta_phowhisper_bilstm.py
"""
Train/Val MER (ViDeBERTa + PhoWhisper + BiLSTM-Attn), export .h5 vào /root/MER/checkpoints,
và chạy đánh giá K-Fold (mặc định K=3, đổi qua env KFOLD).
Hỗ trợ:
- EarlyStopping (EARLY_PATIENCE, default 5)
- Quiet mode để ẩn log training (QUIET=1)
- Override encoder ckpt qua TEXT_ENCODER_CKPT, PHOWHISPER_CKPT
"""

import os
import json
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Subset, DataLoader

# Tái dùng training utils
from train import Trainer, compute_class_weights_lenfreq
from configs.base import Config
from loading.dataloader import VNEMOSDataset, Collator, LengthBucketBatchSampler
from model.networks import MER
from model.losses import CrossEntropyLoss
from training.callbacks import CheckpointsCallback
from training.optimizers import split_param_groups, build_optimizer
from transformers import get_cosine_schedule_with_warmup

# ===== Quiet mode (ẩn log training) =====
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


# ===== Early Stopping callback =====
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

    def __call__(self, trainer: Trainer, logs: dict, isValPhase: bool):
        if not isValPhase or logs is None or self.monitor not in logs:
            return
        val = float(logs[self.monitor])

        improved = False
        if self.best is None:
            improved = True
        else:
            if self.mode == "max":
                improved = (val > self.best + self.min_delta)
            else:
                improved = (val < self.best - self.min_delta)

        if improved:
            self.best = val
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                if not QUIET:
                    print(f"[EarlyStopping] Patience reached ({self.patience}). Stop training.")
                raise _StopTraining()


# ===== Cấu hình mặc định =====
def _build_cfg() -> Config:
    ckpt_root = Path("/root/MER/checkpoints")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_root / "videberta_phowhisper_bilstm"

    text_ckpt = os.getenv("TEXT_ENCODER_CKPT", "videberta-base").strip()
    audio_ckpt = os.getenv("PHOWHISPER_CKPT", "phowhisper-base").strip()

    return Config(
        # run metadata
        name="MER_ViDeBERTa_PhoWhisper_BiLSTM",
        checkpoint_dir=str(ckpt_dir),
        num_epochs=50,
        batch_size=8,
        num_workers=2,

        # optim & scheduler
        learning_rate=2e-4,
        optimizer_type="AdamW",
        scheduler_type="cosine_warmup",
        warmup_ratio=0.05,
        scheduler_step_unit="step",

        # loss
        loss_type="CrossEntropyLoss",
        label_smoothing=0.0,

        # data
        data_root="output",
        jsonl_dir="",
        sample_rate=16000,
        max_audio_sec=None,
        text_max_length=96,

        # sampling
        use_length_bucket=True,
        length_bucket_size=64,
        bucketing_text_alpha=0.03,
        use_weighted_sampler=True,   # chỉ dùng cho train/val theo split sẵn
        lenfreq_alpha=0.5,

        # MODEL
        model_type="MemoCMT",

        # Text encoder → ViDeBERTa
        text_encoder_type="videberta",
        text_encoder_ckpt=text_ckpt,
        text_encoder_dim=768,
        text_unfreeze=False,

        # Audio encoder → PhoWhisper
        audio_encoder_type="phowhisper",
        audio_encoder_ckpt=audio_ckpt,
        audio_encoder_dim=768,
        audio_unfreeze=False,

        # (tuỳ chọn) override FE Whisper
        whisper_n_fft=None,
        whisper_hop_length=None,
        whisper_win_length=None,
        whisper_nb_mels=None,

        # Fusion Bi-LSTM Attention
        fusion_type="bilstm_attn",
        fusion_dim=768,
        fusion_head_output_type="max",   # 'cls' | 'mean' | 'max' | 'min' | 'attn'
        linear_layer_output=[256, 128],
        dropout=0.10,

        fusion_bilstm_hidden_text=384,
        fusion_bilstm_hidden_audio=384,
        fusion_bilstm_layers=1,
        fusion_bilstm_dropout=0.1,
        fusion_bilstm_bidirectional=True,
        fusion_blocks=1,
        fusion_merge="concat",
        fusion_pool_heads=1,

        # Train tricks
        use_amp=True,
        max_grad_norm=1.0,

        save_best_val=True,
        max_to_keep=2,
    )


def _new_trainer_and_optim(cfg: Config, num_training_steps: int) -> Tuple[Trainer, torch.optim.Optimizer, object]:
    device = "cuda" if (torch.cuda.is_available() and os.getenv("USE_CUDA", "1") != "0") else "cpu"
    net = MER(cfg, device=device).to(device)
    trainer = Trainer(cfg, net, CrossEntropyLoss(cfg, weight=None), log_dir="logs")

    enc_lr, head_lr = 5e-6, float(cfg.learning_rate)
    param_groups = split_param_groups(trainer, lr_enc=enc_lr, lr_head=head_lr, weight_decay=0.05)
    optimizer = build_optimizer("adamw", param_groups, lr=head_lr, weight_decay=0.05)

    warmup_steps = max(1, int(cfg.warmup_ratio * num_training_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    trainer.compile(
        optimizer=optimizer,
        scheduler=scheduler,
        lr=head_lr,
        param_groups=None,
        scheduler_step_unit=cfg.scheduler_step_unit
    )
    return trainer, optimizer, scheduler


def _evaluate_and_print(trainer: Trainer, eval_loader, label2id: dict, title: str = "Evaluation"):
    try:
        from sklearn.metrics import classification_report
        HAS_SK = True
    except Exception:
        HAS_SK = False

    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    num_classes = len(id2label)

    all_preds, all_labels = [], []
    trainer.network.eval()
    with torch.no_grad():
        for batch in eval_loader:
            tok, audio, labels, meta = trainer._unpack_batch(batch)
            audio = audio.to(trainer.device)
            labels = labels.to(trainer.device)

            from transformers import BatchEncoding
            if isinstance(tok, BatchEncoding):
                tok = tok.to(trainer.device); tok = {k: v for k, v in tok.items()}
            elif isinstance(tok, dict):
                tok = {k: v.to(trainer.device) for k, v in tok.items()}
            else:
                tok = tok.to(trainer.device)

            out = trainer.network(tok, audio, meta=meta)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for t, p in zip(labels, preds):
        cm[t.long(), p.long()] += 1

    per_class_acc = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()).numpy()
    overall_acc = (preds == labels).float().mean().item()

    print(f"\n=== {title} ===")
    print("Overall Acc: %.4f" % overall_acc)
    print("Per-class Acc:")
    for i, acc in enumerate(per_class_acc):
        name = id2label[i] if i < len(id2label) else str(i)
        print(f"  {i} ({name}): {acc:.4f}")
    print("Confusion Matrix:\n", cm.numpy())

    if HAS_SK:
        target_names = [id2label[i] for i in range(num_classes)]
        print("\nClassification Report:")
        print(classification_report(labels.numpy(), preds.numpy(), target_names=target_names, digits=4))
    else:
        print("sklearn not available: skip classification_report")


def _export_h5(trainer: Trainer, cfg: Config, label2id: dict, out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / filename

    try:
        import h5py
    except Exception:
        print("[WARN] h5py chưa được cài. Cài nhanh:  pip install h5py")
        return

    state = trainer.network.state_dict()
    torch_ver = str(getattr(torch, "__version__", "unknown"))
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    id2label_json = json.dumps(id2label, ensure_ascii=False)

    with h5py.File(h5_path, "w") as f:
        f.attrs["model_name"] = "MER"
        f.attrs["num_classes"] = int(getattr(cfg, "num_classes", len(id2label)))
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


# ===== Helpers cho K-Fold =====
def _make_loader_from_indices(cfg: Config, base_ds: VNEMOSDataset, indices: List[int],
                              shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
    subset = Subset(base_ds, indices)
    collate = Collator(base_ds.tokenizer, text_max_length=getattr(cfg, "text_max_length", 64))

    pin = torch.cuda.is_available()
    nw = max(0, int(getattr(cfg, "num_workers", 0)))
    persistent = True if nw > 0 else False
    bs = int(batch_size or cfg.batch_size)

    # Với KFold ta không dùng WeightedRandomSampler để tránh rối class-weight; cứ shuffle True/False
    # Nếu bạn muốn length-bucket để giảm pad: bucket theo mix_lengths
    if bool(getattr(cfg, "use_length_bucket", False)):
        # Chuẩn bị mix_lengths cho subset
        mix = [base_ds.mix_lengths[i] for i in indices]
        # Map index local của subset -> mix length
        # Dùng sampler custom cho subset: cần một bản sampler nhận list length
        # Ở đây, đơn giản dùng shuffle thường cho an toàn vì subset index không liên tục.
        loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent,
        )
    else:
        loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent,
        )
    return loader


def _compute_class_weights_from_indices(labels: List[int], num_classes: int) -> torch.Tensor:
    import numpy as np
    cnt = np.bincount(labels, minlength=num_classes).astype(float)
    cnt[cnt == 0] = 1.0
    inv = 1.0 / cnt
    w = inv * (num_classes / inv.sum())
    return torch.tensor(w, dtype=torch.float32)


def run_kfold(cfg: Config, K: int = 3, random_state: int = 42):
    """Đánh giá K-Fold trên toàn bộ dữ liệu train.jsonl (gộp train+valid nếu bạn muốn trước đó)."""
    from sklearn.model_selection import StratifiedKFold

    full_ds = VNEMOSDataset(cfg, "train")
    label2id = full_ds.label2id
    y = [label2id[it["emotion"]] for it in full_ds.items]
    num_classes = len(label2id)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)

    fold_accs: List[float] = []
    try:
        from sklearn.metrics import accuracy_score
        HAS_SK = True
    except Exception:
        HAS_SK = False

    # Giảm epoch cho KFold nếu muốn nhanh (tùy chỉnh qua env KFOLD_EPOCHS)
    k_epochs = int(os.getenv("KFOLD_EPOCHS", "20"))

    print(f"\n>>> K-Fold Evaluation (K={K}, epochs={k_epochs})")

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(list(range(len(y))), y), start=1):
        print(f"\n[Fold {fold_id}/{K}] train={len(train_idx)}  val={len(val_idx)}")

        # Dataloader cho fold
        train_loader = _make_loader_from_indices(cfg, full_ds, train_idx, shuffle=True)
        val_loader   = _make_loader_from_indices(cfg, full_ds, val_idx, shuffle=False)

        # Trainer & Optim
        total_steps = len(train_loader) * k_epochs
        trainer, optimizer, scheduler = _new_trainer_and_optim(cfg, num_training_steps=total_steps)

        # Class weights theo subset train của fold
        w = _compute_class_weights_from_indices([y[i] for i in train_idx], num_classes).to(trainer.device)
        trainer.criterion = CrossEntropyLoss(cfg, weight=w)

        # Callbacks (có early stop)
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

        # Train fold
        try:
            trainer.fit(train_loader, epochs=k_epochs, eval_data=val_loader, callbacks=callbacks)
        except _StopTraining:
            if not QUIET:
                print(f"[Fold {fold_id}] Early stopped.")

        # Load best (nếu có)
        best_path = getattr(ckpt_cb, "best_path", "")
        if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
            state = torch.load(best_path, map_location=trainer.device)
            trainer.network.load_state_dict(state)

        # Inference fold
        preds, labels = [], []
        trainer.network.eval()
        with torch.no_grad():
            for batch in val_loader:
                tok, audio, labs, meta = trainer._unpack_batch(batch)
                audio = audio.to(trainer.device)
                from transformers import BatchEncoding
                if isinstance(tok, BatchEncoding):
                    tok = tok.to(trainer.device); tok = {k: v for k, v in tok.items()}
                elif isinstance(tok, dict):
                    tok = {k: v.to(trainer.device) for k, v in tok.items()}
                else:
                    tok = tok.to(trainer.device)

                out = trainer.network(tok, audio, meta=meta)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                p = torch.argmax(logits, dim=1)
                preds.append(p.cpu()); labels.append(labs.cpu())

        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()

        if HAS_SK:
            acc = accuracy_score(labels, preds)
        else:
            acc = float((preds == labels).mean())

        fold_accs.append(acc)
        print(f"[Fold {fold_id}] Acc = {acc:.4f}")

    if fold_accs:
        import numpy as np
        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        print(f"\n>>> K-Fold (K={K}) Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print("\n>>> K-Fold: no folds computed.")


def main():
    # === Cấu hình ===
    cfg = _build_cfg()

    # === Data (Train/Val theo file split có sẵn) ===
    # Tái dùng builder chuẩn của bạn
    from loading.dataloader import build_train_test_dataset
    train_loader, eval_loader = build_train_test_dataset(cfg)

    # Lấy nhãn từ train set
    _train_ds = VNEMOSDataset(cfg, "train")
    label2id = _train_ds.label2id
    cfg.num_classes = len(label2id)

    # === Trainer + Optim ===
    total_steps = len(train_loader) * cfg.num_epochs
    trainer, optimizer, scheduler = _new_trainer_and_optim(cfg, num_training_steps=total_steps)

    # === Class weights (len-freq) ===
    class_weights = compute_class_weights_lenfreq(_train_ds, alpha=float(getattr(cfg, "lenfreq_alpha", 0.5))).to(trainer.device)
    if not QUIET:
        print("Class weights (len-freq):", class_weights.tolist())
    trainer.criterion = CrossEntropyLoss(cfg, weight=class_weights)

    # === Callbacks (checkpoint + early stop) ===
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

    # === Train/Val ===
    try:
        trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=eval_loader, callbacks=callbacks)
    except _StopTraining:
        if not QUIET:
            print("Training stopped early by EarlyStopping.")

    best_path = getattr(ckpt_cb, "best_path", "")
    if not QUIET:
        print("Best checkpoint:", best_path)

    # Load best weights (nếu có)
    if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
        if not QUIET:
            print("Loading best weights:", best_path)
        state = torch.load(best_path, map_location=trainer.device)
        trainer.network.load_state_dict(state)

    # === Evaluate & Report ===
    if eval_loader is not None:
        _evaluate_and_print(trainer, eval_loader, label2id, title="Evaluation")
    else:
        print("No eval_loader available.")

    # === Export HDF5 (.h5) vào /root/MER/checkpoints ===
    _export_h5(
        trainer,
        cfg,
        label2id,
        out_dir=Path("/root/MER/checkpoints"),
        filename="model_weights_videberta_phowhisper_bilstm.h5",
    )

    # === K-Fold Evaluation (mặc định K=3, đổi bằng env KFOLD) ===
    k = int(os.getenv("KFOLD", "3"))
    run_kfold(cfg, K=k, random_state=42)


if __name__ == "__main__":
    main()
