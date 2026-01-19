# src/train_mer_videberta_fourier_bilstm.py
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Iterable, List

USE_CUDA_ENV = os.getenv("USE_CUDA", "1").strip()
FORCE_CPU = (USE_CUDA_ENV == "0")

import torch
from torch import Tensor
from transformers import BatchEncoding, get_cosine_schedule_with_warmup

from configs.base import Config
from model.networks import MER
from model.losses import CrossEntropyLoss
from training.trainer import TorchTrainer
from loading.dataloader import build_train_test_dataset, VNEMOSDataset
from training.callbacks import CheckpointsCallback
from training.optimizers import split_param_groups, build_optimizer


# ========= Path helpers (relative & portable) =========
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent  # repo root (.. from src/)
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"
# ViDeBERTa mirror mặc định nằm ở PROJECT_ROOT / "videberta-base"
VIDEBERTA_DIR = PROJECT_ROOT / "videberta-base"  # có thể thay bằng HF id nếu không có mirror local


# =========================
# Helpers
# =========================
def _to_device_text(x, device):
    if isinstance(x, BatchEncoding):
        x = x.to(device)
        return {k: v for k, v in x.items()}
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def compute_class_weights_lenfreq(train_ds, alpha: float = 0.5):
    import numpy as np
    label_ids, lengths = [], []
    for it in train_ds.items:
        c = train_ds.label2id[it["emotion"]]
        L = int(round((float(it["end"]) - float(it["start"])) * train_ds.sample_rate))
        label_ids.append(c); lengths.append(max(L, 1))
    label_ids = np.asarray(label_ids, dtype=np.int64)
    lengths   = np.asarray(lengths,   dtype=np.float64)

    C = len(train_ds.label2id)
    freq = np.zeros(C, dtype=np.float64)
    len_mean = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (label_ids == c)
        freq[c] = max(mask.sum(), 1)
        len_mean[c] = lengths[mask].mean() if mask.any() else lengths.mean()

    inv_freq = (freq.mean() / freq)
    inv_len  = (len_mean.mean() / len_mean) ** alpha
    w = inv_freq * inv_len
    w = w * (C / w.sum())
    return torch.tensor(w, dtype=torch.float32)


class Trainer(TorchTrainer):
    def __init__(self, cfg: Config, network: MER, criterion: torch.nn.Module = None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion

        if FORCE_CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.use_amp = (self.device.type == "cuda") and bool(getattr(cfg, "use_amp", torch.cuda.is_available()))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))

    @staticmethod
    def _to_float(v) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        if torch.is_tensor(v):
            if v.ndim == 0: return float(v.item())
            return None
        try:
            import numpy as _np
            if isinstance(v, _np.generic):
                return float(v)
        except Exception:
            pass
        return None

    @classmethod
    def _combine_logs_mean(cls, logs_list: List[Dict[str, object]]) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        if not logs_list: return agg
        keys = set()
        for d in logs_list: keys.update(d.keys())
        for k in keys:
            vals: List[float] = []
            for x in logs_list:
                if k in x:
                    f = cls._to_float(x[k])
                    if f is not None: vals.append(f)
            if vals: agg[k] = sum(vals) / len(vals)
        return agg

    @staticmethod
    def _compute_macro_f1_from_preds(preds: torch.Tensor, targets: torch.Tensor) -> float:
        num_classes = int(max(int(preds.max().item()), int(targets.max().item())) + 1)
        f1s = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return float(sum(f1s) / len(f1s)) if f1s else 0.0

    def _unpack_batch(self, batch: Union[Tuple, Tuple[Tuple, Dict]]):
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            data, meta = batch
            return data[0], data[1], data[2], meta
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], dict):
            tok, rest = batch
            return tok, rest[0], rest[1], None
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            return batch[0], batch[1], batch[2], None
        if isinstance(batch, dict):
            return batch.get("text"), batch.get("audio"), batch.get("label"), None
        raise ValueError(f"Batch structure không hỗ trợ: preview={str(batch)[:200]}")

    def train_step(self, batch) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        input_text, input_audio, label, meta = self._unpack_batch(batch)
        input_audio = input_audio.to(self.device, non_blocking=(self.device.type == "cuda"))
        label = label.to(self.device, non_blocking=(self.device.type == "cuda"))
        input_text = _to_device_text(input_text, self.device)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                logits, *_ = self.network(input_text, input_audio, meta=meta)
                loss = self.criterion(logits, label)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer); self.scaler.update()
        else:
            logits, *_ = self.network(input_text, input_audio, meta=meta)
            loss = self.criterion(logits, label)
            loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()

        return {"loss": float(loss.detach().cpu()), "acc": float(acc.detach().cpu())}

    def test_step(self, batch) -> Dict[str, float]:
        self.network.eval()
        input_text, input_audio, label, meta = self._unpack_batch(batch)
        input_audio = input_audio.to(self.device, non_blocking=(self.device.type == "cuda"))
        label = label.to(self.device, non_blocking=(self.device.type == "cuda"))
        input_text = _to_device_text(input_text, self.device)
        with torch.no_grad():
            logits, *_ = self.network(input_text, input_audio, meta=meta)
            loss = self.criterion(logits, label)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).float().mean()
        return {"val_loss": float(loss.detach().cpu()),
                "val_acc": float(acc.detach().cpu()),
                "preds": preds.detach().cpu(),
                "targets": label.detach().cpu()}

    def fit(self,
            train_loader: Iterable,
            epochs: int = 1,
            eval_data: Optional[Iterable] = None,
            callbacks: Optional[List[object]] = None):
        callbacks = callbacks or []
        self.global_step = 0
        self.global_epoch = 0

        for epoch in range(1, epochs + 1):
            self.global_epoch = epoch
            self.network.train()
            epoch_logs = []

            for batch in train_loader:
                step_logs = self.train_step(batch)
                self.global_step += 1
                epoch_logs.append(step_logs)

                if self.scheduler is not None and self.scheduler_step_unit == "step":
                    try: self.scheduler.step()
                    except Exception: pass

                logs_for_cb = dict(step_logs)
                for cb in callbacks:
                    try: cb(self, self.global_step, self.global_epoch, logs_for_cb, isValPhase=False, logger=self.logger)
                    except Exception as e: self.logger.warning("Callback error (train step): %s", e)

            mean_logs = self._combine_logs_mean(epoch_logs)
            self.logger.info(f"[Epoch {epoch}] train: {mean_logs}")

            if eval_data is not None:
                self.network.eval()
                vlogs = []; all_preds, all_targets = [], []
                with torch.no_grad():
                    for batch in eval_data:
                        out = self.test_step(batch)
                        vlogs.append(out)
                        if "preds" in out and "targets" in out:
                            all_preds.append(out["preds"]); all_targets.append(out["targets"])

                val_logs = self._combine_logs_mean(vlogs)
                if all_preds and all_targets:
                    preds = torch.cat(all_preds); targs = torch.cat(all_targets)
                    try:
                        from sklearn.metrics import f1_score
                        val_macro_f1 = float(f1_score(targs.numpy(), preds.numpy(), average="macro"))
                    except Exception:
                        val_macro_f1 = self._compute_macro_f1_from_preds(preds, targs)
                    val_logs["val_macro_f1"] = val_macro_f1

                self.logger.info(f"[Epoch {epoch}] val:   {val_logs}")

                for cb in callbacks:
                    try: cb(self, self.global_step, self.global_epoch, val_logs, isValPhase=True, logger=self.logger)
                    except Exception as e: self.logger.warning("Callback error (val): %s", e)

            if self.scheduler is not None and self.scheduler_step_unit == "epoch":
                try: self.scheduler.step()
                except Exception: pass
        return True


# =========================
# Main
# =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Cấu hình 3 thành phần:
    #   - TEXT  : ViDeBERTa (mirror local tại PROJECT_ROOT/videberta-base)
    #   - AUDIO : Fourier2Vec
    #   - FUSION: Bi-LSTM Attention
    ckpt_dir = CHECKPOINT_ROOT / "mer_videberta_fourier_bilstm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        name="MER_ViDeBERTa_Fourier2Vec_BiLSTM",
        checkpoint_dir=str(ckpt_dir),
        num_epochs=30,
        batch_size=8,
        num_workers=2,

        # Optim & Scheduler
        learning_rate=2e-5,
        optimizer_type="AdamW",
        scheduler_type="cosine_warmup",
        warmup_ratio=0.05,
        scheduler_step_unit="step",

        # Loss
        loss_type="CrossEntropyLoss",
        label_smoothing=0.0,

        # Data (đường dẫn tương đối)
        data_root=str(OUTPUT_DIR),
        jsonl_dir="",
        sample_rate=16000,
        max_audio_sec=None,
        text_max_length=96,

        # Sampler/bucketing
        use_length_bucket=True,
        length_bucket_size=64,
        bucketing_text_alpha=0.03,
        use_weighted_sampler=True,
        lenfreq_alpha=0.5,

        # MODEL core
        model_type="MemoCMT",

        # --- TEXT: ViDeBERTa ---
        text_encoder_type="videberta",
        text_encoder_ckpt=str(VIDEBERTA_DIR if VIDEBERTA_DIR.exists() else "Fsoft-AIC/videberta-base"),
        text_encoder_dim=768,
        text_unfreeze=False,

        # --- AUDIO: Fourier2Vec ---
        audio_encoder_type="fourier2vec",
        audio_encoder_dim=256,   # sẽ được overwrite bởi encoder.config.hidden_size nếu có
        audio_unfreeze=False,

        # Fourier2Vec hyper (khớp modules đã cập nhật)
        fourier_n_mels=64,
        fourier_fmin=125.0,
        fourier_fmax=7500.0,
        fourier_win_ms=25.0,
        fourier_hop_ms=10.0,
        fourier_patch_len=1,
        fourier_patch_hop=1,
        fourier_hidden_size=256,
        fourier_num_heads=4,
        fourier_num_layers=4,

        # --- FUSION: Bi-LSTM Attention ---
        fusion_type="bilstm_attn",
        fusion_dim=768,
        fusion_bilstm_hidden_text=384,
        fusion_bilstm_hidden_audio=384,
        fusion_bilstm_layers=1,
        fusion_bilstm_dropout=0.1,
        fusion_bilstm_bidirectional=True,
        fusion_blocks=1,
        fusion_merge="concat",          # hoặc "gate"
        fusion_head_output_type="attn",
        fusion_pool_heads=1,

        # Head & regularization
        linear_layer_output=[256, 128],
        dropout=0.10,

        # AMP & grad
        use_amp=True,
        max_grad_norm=1.0,

        # Checkpointing
        save_best_val=True,
        max_to_keep=2,
    )

    # ===== Dataloader & label space =====
    train_loader, eval_loader = build_train_test_dataset(cfg)
    _train_ds = VNEMOSDataset(cfg, "train")
    label2id = _train_ds.label2id
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    cfg.num_classes = len(id2label)

    # ===== Network =====
    net = MER(cfg, device="cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu")
    net = net.to("cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu")

    # ===== Class weights (len-freq) =====
    class_weights = compute_class_weights_lenfreq(_train_ds, alpha=0.5).to(
        "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
    )
    print("Class weights (len-freq):", class_weights.tolist())
    criterion = CrossEntropyLoss(cfg, weight=class_weights)

    trainer = Trainer(cfg, net, criterion, log_dir=str(PROJECT_ROOT / "logs"))

    # ===== LR param groups: encoder nhỏ, head/fusion lớn =====
    enc_lr, head_lr = 5e-6, 2e-4
    param_groups = split_param_groups(trainer, lr_enc=enc_lr, lr_head=head_lr, weight_decay=0.05)
    optimizer = build_optimizer("adamw", param_groups, lr=head_lr, weight_decay=0.05)

    # ===== Scheduler cosine + warmup =====
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

    # ===== Checkpoint callback (best by macro-F1) =====
    ckpt_cb = CheckpointsCallback(
        cfg.checkpoint_dir,
        save_freq=200,
        max_to_keep=cfg.max_to_keep,
        save_best_val=True,
        monitor="val_macro_f1",
        mode="max",
    )
    callbacks = [ckpt_cb]

    # ===== Train =====
    trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=eval_loader, callbacks=callbacks)
    best_path = getattr(ckpt_cb, "best_path", "")
    print("Best checkpoint:", best_path)

    # ===== (Optional) Load best weights =====
    if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
        print("Loading best weights:", best_path)
        state = torch.load(best_path, map_location=trainer.device)
        trainer.network.load_state_dict(state)


    def collect_preds(_trainer, loader):
        all_preds, all_labels = [], []
        _trainer.network.eval()
        with torch.no_grad():
            for batch in loader:
                (tok, audio, labels, meta) = _trainer._unpack_batch(batch)
                audio  = audio.to(_trainer.device)
                labels = labels.to(_trainer.device)
                tok    = _to_device_text(tok, _trainer.device)
                out = _trainer.network(tok, audio, meta=meta)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu()); all_labels.append(labels.cpu())
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

    # ===== Export HDF5 weights (giống src/train.py) =====
    import h5py, json
    h5_name = "model_weights_best.h5"
    h5_path = Path(cfg.checkpoint_dir) / h5_name

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
    print(f"[+] Saved HDF5 weights to: {h5_path.relative_to(PROJECT_ROOT)}")
