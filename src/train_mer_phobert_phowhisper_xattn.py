# -*- coding: utf-8 -*-
# src/train_mer_phobert_phowhisper_xattn.py

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Subset, DataLoader

# Reuse Trainer + helpers from src/train.py
from train import Trainer, compute_class_weights_lenfreq
from configs.base import Config
from loading.dataloader import (
    VNEMOSDataset,
    Collator,
    LengthBucketBatchSampler,
)
from model.networks import MER
from model.losses import CrossEntropyLoss
from training.callbacks import CheckpointsCallback
from training.optimizers import split_param_groups, build_optimizer
from transformers import get_cosine_schedule_with_warmup

try:
    from sklearn.metrics import classification_report
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# =========================
# Helpers
# =========================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _unpack_batch_compat(trainer: Trainer, batch):
    """
    Một số phiên bản Trainer có _unpack_batch, số khác có _standardize_batch.
    Dùng hàm tương thích để tránh crash.
    """
    if hasattr(trainer, "_unpack_batch"):
        return trainer._unpack_batch(batch)  # type: ignore
    return trainer._standardize_batch(batch)


def _build_cfg() -> Config:
    """
    Cấu hình PhoBERT + PhoWhisper + Cross-Attention (XAttn)
    - checkpoint_dir: /root/MER/checkpoints/phobert_phowhisper_xattn
    - audio_encoder_ckpt: đường dẫn local như yêu cầu.
    """
    ckpt_root = Path("/root/MER/checkpoints")
    _ensure_dir(ckpt_root)
    ckpt_dir = ckpt_root / "phobert_phowhisper_xattn"

    cfg = Config(
        # run meta
        name="MER_PhoBERT_PhoWhisper_XAttn",
        checkpoint_dir=str(ckpt_dir),
        num_epochs=50,
        batch_size=8,
        num_workers=2,

        # tối ưu
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
        max_audio_sec=None,  # PhoWhisper tự truncate theo ~30s nếu cần
        text_max_length=96,

        # sampling
        use_length_bucket=True,
        length_bucket_size=64,
        bucketing_text_alpha=0.03,
        use_weighted_sampler=True,
        lenfreq_alpha=0.5,

        # MODEL
        model_type="MemoCMT",

        # Text encoder → PhoBERT
        text_encoder_type="phobert",
        text_encoder_ckpt="vinai/phobert-base",
        text_encoder_dim=768,
        text_unfreeze=False,

        # Audio encoder → PhoWhisper (đường dẫn local)
        audio_encoder_type="phowhisper",
        audio_encoder_ckpt="/root/MER/phowhisper-base",
        audio_encoder_dim=768,  # sẽ bị overwrite bởi encoder.config.hidden_size
        audio_unfreeze=False,

        # (tuỳ chọn) override tham số feature extractor
        whisper_n_fft=None,
        whisper_hop_length=None,
        whisper_win_length=None,
        whisper_nb_mels=None,

        # Fusion = Cross-Attention
        fusion_type="xattn",
        fusion_dim=768,
        fusion_head_output_type="max",   # 'cls'|'mean'|'max'|'min'|'attn'
        linear_layer_output=[256, 128],
        dropout=0.10,

        # BiLSTM params không dùng ở xattn nhưng vẫn có trong cfg
        fusion_bilstm_hidden_text=384,
        fusion_bilstm_hidden_audio=384,
        fusion_bilstm_layers=1,
        fusion_bilstm_dropout=0.1,
        fusion_bilstm_bidirectional=True,
        fusion_blocks=1,
        fusion_merge="concat",
        fusion_pool_heads=1,

        # Tricks
        use_amp=True,
        max_grad_norm=1.0,

        save_best_val=True,
        max_to_keep=2,
    )
    return cfg


def _new_trainer_and_optim(cfg: Config, num_training_steps: int) -> Tuple[Trainer, torch.optim.Optimizer, object]:
    """
    Khởi tạo MER + Trainer + Optim + Scheduler (cosine warmup).
    """
    device = "cuda" if (torch.cuda.is_available() and os.getenv("USE_CUDA", "1") != "0") else "cpu"
    net = MER(cfg, device=device).to(device)

    # tạm thời đặt criterion, sẽ gán lại với class-weights thực từ train-set
    trainer = Trainer(cfg, net, CrossEntropyLoss(cfg, weight=None), log_dir="logs")

    # Optim với param groups: LR encoder nhỏ, head lớn hơn
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
        scheduler_step_unit=cfg.scheduler_step_unit,
    )
    return trainer, optimizer, scheduler


def _evaluate_and_print(trainer: Trainer, eval_loader, label2id: dict, title: str = "Evaluation"):
    """
    Chạy forward trên eval_loader và in các chỉ số, confusion matrix, classification report.
    Trả về: (overall_acc: float, per_class_acc: np.ndarray, cm: torch.Tensor[int32])
    """
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    num_classes = len(id2label)

    all_preds, all_labels = [], []
    trainer.network.eval()
    with torch.no_grad():
        for batch in eval_loader:
            tok, audio, labels, meta = _unpack_batch_compat(trainer, batch)
            audio = audio.to(trainer.device)
            labels = labels.to(trainer.device)

            from transformers import BatchEncoding
            if isinstance(tok, BatchEncoding):
                tok = tok.to(trainer.device)
                tok = {k: v for k, v in tok.items()}
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

    if _HAS_SK:
        target_names = [id2label[i] for i in range(num_classes)]
        print("\nClassification Report:")
        print(classification_report(labels.numpy(), preds.numpy(), target_names=target_names, digits=4))
    else:
        print("sklearn not available: skip classification_report")

    return overall_acc, per_class_acc, cm  # cm là torch.Tensor


def _export_h5(trainer: Trainer, cfg: Config, label2id: dict, out_dir: Path, filename: str):
    """
    Xuất weights .h5 vào out_dir/filename, ghi kèm metadata & config.
    """
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


# =========================
# Train/Val theo split có sẵn
# =========================
def run_train_val(cfg: Config):
    """
    Train theo split có sẵn (train + valid|test), lưu best checkpoint
    và export .h5 vào /root/MER/checkpoints
    """
    base_dir = Path(cfg.data_root) / (getattr(cfg, "jsonl_dir", "") or "")
    eval_split = "valid" if (base_dir / "valid.jsonl").exists() else "test"

    train_set = VNEMOSDataset(cfg, "train")
    label2id = train_set.label2id
    cfg.num_classes = len(label2id)
    collate = Collator(train_set.tokenizer, text_max_length=getattr(cfg, "text_max_length", 64))

    # Sampler cho train
    pin = torch.cuda.is_available()
    nw = max(0, int(getattr(cfg, "num_workers", 0)))
    persistent = True if nw > 0 else False
    use_len_bucket = bool(getattr(cfg, "use_length_bucket", False))
    bucket_size = int(getattr(cfg, "length_bucket_size", 64))

    if use_len_bucket:
        mix_lengths = [train_set.mix_lengths[i] for i in range(len(train_set))]
        train_sampler = LengthBucketBatchSampler(
            lengths=mix_lengths,
            batch_size=cfg.batch_size,
            shuffle=True,
            bucket_size=bucket_size,
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent,
        )
    else:
        if bool(getattr(cfg, "use_weighted_sampler", True)):
            from torch.utils.data import WeightedRandomSampler
            wrs = WeightedRandomSampler(
                weights=torch.tensor(train_set.sample_weights_lenfreq, dtype=torch.float),
                num_samples=len(train_set),
                replacement=True,
            )
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.batch_size,
                sampler=wrs,
                num_workers=nw,
                collate_fn=collate,
                pin_memory=pin,
                persistent_workers=persistent,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=nw,
                collate_fn=collate,
                pin_memory=pin,
                persistent_workers=persistent,
            )

    eval_set = VNEMOSDataset(cfg, eval_split, label2id)
    eval_loader = DataLoader(
        eval_set,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    # Trainer + Optim + Scheduler
    total_steps = len(train_loader) * cfg.num_epochs
    trainer, optimizer, scheduler = _new_trainer_and_optim(cfg, num_training_steps=total_steps)

    # Class weights từ train_set
    device = trainer.device
    class_weights = compute_class_weights_lenfreq(train_set, alpha=float(getattr(cfg, "lenfreq_alpha", 0.5))).to(device)
    print("Class weights (len-freq):", class_weights.tolist())
    trainer.criterion = CrossEntropyLoss(cfg, weight=class_weights)

    # Callbacks (best theo val_macro_f1)
    ckpt_cb = CheckpointsCallback(
        cfg.checkpoint_dir,
        save_freq=200,
        max_to_keep=cfg.max_to_keep,
        save_best_val=True,
        monitor="val_macro_f1",
        mode="max",
    )
    callbacks = [ckpt_cb]

    # Train
    trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=eval_loader, callbacks=callbacks)
    best_path = getattr(ckpt_cb, "best_path", "")
    print("Best checkpoint:", best_path)

    # (Optional) load best
    if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
        print("Loading best weights:", best_path)
        state = torch.load(best_path, map_location=trainer.device)
        trainer.network.load_state_dict(state)

    # Evaluate & export .h5
    _evaluate_and_print(trainer, eval_loader, label2id, title="Evaluation (Holdout)")
    _export_h5(
        trainer,
        cfg,
        label2id,
        out_dir=Path("/root/MER/checkpoints"),
        filename="model_weights_phobert_phowhisper_xattn.h5",
    )


# =========================
# K-Fold (mặc định K=5)
# =========================
def run_kfold(cfg: Config, K: int = 5, random_state: int = 42):
    """
    Đánh giá K-Fold trên dữ liệu TRAIN JSONL (bỏ qua valid/test có sẵn).
    Mỗi fold: train trên (K-1) phần, validate trên 1 phần.
    """
    from sklearn.model_selection import KFold
    print(f"\n========== K-Fold Evaluation (K={K}) ==========")

    full = VNEMOSDataset(cfg, "train")  # dùng toàn bộ train.jsonl cho KFold
    label2id = full.label2id
    cfg.num_classes = len(label2id)

    collate = Collator(full.tokenizer, text_max_length=getattr(cfg, "text_max_length", 64))

    pin = torch.cuda.is_available()
    nw = max(0, int(getattr(cfg, "num_workers", 0)))
    persistent = True if nw > 0 else False

    # Dùng độ dài mix để bucket
    mix_lengths = full.mix_lengths
    indices = list(range(len(full)))

    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)

    fold_overall_acc: List[float] = []
    fold_cms: List[torch.Tensor] = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n--- Fold {fold_id}/{K} ---")

        train_subset = Subset(full, train_idx.tolist())
        val_subset = Subset(full, val_idx.tolist())

        # Build train loader (bucket theo mix_lengths)
        if bool(getattr(cfg, "use_length_bucket", True)):
            lengths_fold = [mix_lengths[i] for i in train_idx.tolist()]
            sampler = LengthBucketBatchSampler(
                lengths=lengths_fold,
                batch_size=cfg.batch_size,
                shuffle=True,
                bucket_size=int(getattr(cfg, "length_bucket_size", 64)),
            )
            train_loader = DataLoader(
                train_subset,
                batch_sampler=sampler,
                num_workers=nw,
                collate_fn=collate,
                pin_memory=pin,
                persistent_workers=persistent,
            )
        else:
            train_loader = DataLoader(
                train_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=nw,
                collate_fn=collate,
                pin_memory=pin,
                persistent_workers=persistent,
            )

        val_loader = DataLoader(
            val_subset,
            batch_size=max(1, cfg.batch_size),
            shuffle=False,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent,
        )

        # Trainer cho fold này
        steps = len(train_loader) * cfg.num_epochs
        trainer, optimizer, scheduler = _new_trainer_and_optim(cfg, num_training_steps=steps)

        # class weights tính theo *full train set* (ổn), hoặc bạn có thể tính theo subset train_idx nếu muốn
        cw = compute_class_weights_lenfreq(full, alpha=float(getattr(cfg, "lenfreq_alpha", 0.5))).to(trainer.device)
        trainer.criterion = CrossEntropyLoss(cfg, weight=cw)

        # Checkpoint cho fold (lưu vào thư mục fold riêng)
        fold_ckpt_dir = Path(cfg.checkpoint_dir) / f"fold_{fold_id}"
        _ensure_dir(fold_ckpt_dir)
        ckpt_cb = CheckpointsCallback(
            str(fold_ckpt_dir),
            save_freq=200,
            max_to_keep=1,
            save_best_val=True,
            monitor="val_macro_f1",
            mode="max",
        )

        trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=val_loader, callbacks=[ckpt_cb])

        # Load best của fold nếu có
        best_path = getattr(ckpt_cb, "best_path", "")
        if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
            state = torch.load(best_path, map_location=trainer.device)
            trainer.network.load_state_dict(state)

        # Evaluate fold
        overall_acc, per_class_acc, cm = _evaluate_and_print(
            trainer, val_loader, label2id, title=f"Fold {fold_id} Validation"
        )
        fold_overall_acc.append(overall_acc)

        # cm là torch.Tensor → lưu trực tiếp
        if isinstance(cm, torch.Tensor):
            fold_cms.append(cm.to(dtype=torch.int64))
        else:
            import numpy as np
            fold_cms.append(torch.from_numpy(np.asarray(cm)).to(dtype=torch.int64))

    # Tổng hợp KFold
    import numpy as np
    mean_acc = float(np.mean(fold_overall_acc)) if fold_overall_acc else 0.0
    std_acc = float(np.std(fold_overall_acc)) if fold_overall_acc else 0.0
    print("\n========== K-Fold Summary ==========")
    print(f"Acc (mean ± std) over {K} folds: {mean_acc:.4f} ± {std_acc:.4f}")

    # Gộp confusion matrices
    if fold_cms:
        cm_sum = torch.stack(fold_cms, dim=0).sum(dim=0).cpu().numpy()
        print("Summed Confusion Matrix over folds:\n", cm_sum)


# =========================
# Entry
# =========================
def main():
    # 1) Train/Val theo split có sẵn và export .h5 vào /root/MER/checkpoints
    cfg = _build_cfg()
    run_train_val(cfg)

    # 2) Luôn chạy KFold evaluation với K=5 mặc định (theo yêu cầu)
    run_kfold(cfg, K=5, random_state=42)


if __name__ == "__main__":
    # Cho phép override nhanh bằng biến môi trường:
    #   KFOLD=0 -> chỉ Train/Val
    #   KFOLD>0 -> chạy KFold với số fold đó sau Train/Val
    kfold_env = os.getenv("KFOLD", "").strip()
    if kfold_env != "":
        try:
            k = int(kfold_env)
        except ValueError:
            k = 5
        cfg = _build_cfg()
        run_train_val(cfg)
        if k > 0:
            run_kfold(cfg, K=k, random_state=42)
    else:
        main()
