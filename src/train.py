# src/train.py
import os
import logging
from typing import Dict, Tuple, Union, Optional, Iterable, List

# --- Cho phép ép CPU qua biến môi trường (nếu thật sự cần) ---
USE_CUDA_ENV = os.getenv("USE_CUDA", "1").strip()
FORCE_CPU = (USE_CUDA_ENV == "0")

import torch
from torch import Tensor
from transformers import BatchEncoding

from configs.base import Config
from model.networks import MER   # (MemoCMT = MER)
from model.losses import CrossEntropyLoss  # dùng CE có trọng số
from training.trainer import TorchTrainer  # kế thừa; ta sẽ override fit() an toàn


def _to_device_text(x, device):
    # BatchEncoding để encoder nhận **kwargs (giữ attention_mask)
    if isinstance(x, BatchEncoding):
        x = x.to(device)
        return {k: v for k, v in x.items()}
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


class Trainer(TorchTrainer):
    """
    Trainer dùng riêng cho script này:
    - train_step/test_step giữ nguyên logic huấn luyện
    - **Override fit()** để tổng hợp log an toàn (chỉ scalar) + tính val_macro_f1
      → tránh lỗi "only one element tensors can be converted to Python scalars"
    - Dùng AMP API mới của PyTorch.
    """
    def __init__(
        self,
        cfg: Config,
        network: MER,
        criterion: torch.nn.Module = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion

        # --- chọn device: cho phép ép CPU qua USE_CUDA=0 ---
        if FORCE_CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        # --- AMP: dùng API mới của PyTorch ---
        self.use_amp = (self.device.type == "cuda") and bool(getattr(cfg, "use_amp", torch.cuda.is_available()))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)  # API mới

        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))

    # ========= Helpers cho fit() tự tổng hợp logs =========
    @staticmethod
    def _to_float(v) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        if torch.is_tensor(v):
            if v.ndim == 0:
                return float(v.item())
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
        """
        Gộp trung bình các khóa có giá trị scalar-like.
        Bỏ qua các khóa non-scalar (tensor nhiều phần tử, list, v.v.).
        """
        agg: Dict[str, float] = {}
        if not logs_list:
            return agg

        keys = set()
        for d in logs_list:
            keys.update(d.keys())

        for k in keys:
            vals: List[float] = []
            for x in logs_list:
                if k in x:
                    f = cls._to_float(x[k])
                    if f is not None:
                        vals.append(f)
            if vals:
                agg[k] = sum(vals) / len(vals)
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

    # ========= unpack + steps =========
    def _unpack_batch(self, batch: Union[Tuple, Tuple[Tuple, Dict]]):
        """
        Hỗ trợ các biến thể batch do DataLoader trả về:
          - ((tok, audio, labels), meta)
          - (tok, (audio, labels))
          - (tok, audio, labels)
          - dict("text","audio","label")
        """
        # ((tok, audio, labels), meta)
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            data, meta = batch
            return data[0], data[1], data[2], meta

        # (tok, (audio, labels))
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], dict):
            tok, rest = batch
            return tok, rest[0], rest[1], None

        # (tok, audio, labels)
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            return batch[0], batch[1], batch[2], None

        # dict({"text","audio","label"})
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
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, *_ = self.network(input_text, input_audio, meta=meta)
            loss = self.criterion(logits, label)
            loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()

        return {
            "loss": float(loss.detach().cpu()),
            "acc": float(acc.detach().cpu()),
        }

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

        # Trả thêm preds/targets để tính macro-F1 trong fit() override của file này
        return {
            "val_loss": float(loss.detach().cpu()),
            "val_acc": float(acc.detach().cpu()),
            "preds": preds.detach().cpu(),
            "targets": label.detach().cpu(),
        }

    # ========= override fit() để tổng hợp logs an toàn =========
    def fit(
        self,
        train_loader: Iterable,
        epochs: int = 1,
        eval_data: Optional[Iterable] = None,
        callbacks: Optional[List[object]] = None,
    ):
        callbacks = callbacks or []
        self.global_step = 0
        self.global_epoch = 0

        for epoch in range(1, epochs + 1):
            self.global_epoch = epoch
            self.network.train()
            epoch_logs = []

            for batch in train_loader:
                step_logs = self.train_step(batch)  # subclass
                self.global_step += 1
                epoch_logs.append(step_logs)

                # Step scheduler per-step
                if self.scheduler is not None and self.scheduler_step_unit == "step":
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                # Callbacks per step (non-val)
                logs_for_cb = dict(step_logs)  # chỉ scalars
                for cb in callbacks:
                    try:
                        cb(self, self.global_step, self.global_epoch, logs_for_cb, isValPhase=False, logger=self.logger)
                    except Exception as e:
                        self.logger.warning("Callback error (train step): %s", e)

            # Aggregate epoch logs
            mean_logs = self._combine_logs_mean(epoch_logs)
            self.logger.info(f"[Epoch {epoch}] train: {mean_logs}")

            # Validation
            if eval_data is not None:
                self.network.eval()
                vlogs = []
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for batch in eval_data:
                        out = self.test_step(batch)  # subclass
                        vlogs.append(out)
                        if "preds" in out and "targets" in out:
                            all_preds.append(out["preds"])
                            all_targets.append(out["targets"])

                val_logs = self._combine_logs_mean(vlogs)  # mean cho các scalar keys
                # Macro-F1 từ preds/targets nếu có
                if all_preds and all_targets:
                    preds = torch.cat(all_preds)
                    targs = torch.cat(all_targets)
                    try:
                        from sklearn.metrics import f1_score
                        val_macro_f1 = float(f1_score(targs.numpy(), preds.numpy(), average="macro"))
                    except Exception:
                        val_macro_f1 = self._compute_macro_f1_from_preds(preds, targs)
                    val_logs["val_macro_f1"] = val_macro_f1

                self.logger.info(f"[Epoch {epoch}] val:   {val_logs}")

                # Callbacks per validation phase (chỉ scalars trong val_logs)
                for cb in callbacks:
                    try:
                        cb(self, self.global_step, self.global_epoch, val_logs, isValPhase=True, logger=self.logger)
                    except Exception as e:
                        self.logger.warning("Callback error (val): %s", e)

            # Step scheduler per-epoch
            if self.scheduler is not None and self.scheduler_step_unit == "epoch":
                try:
                    self.scheduler.step()
                except Exception:
                    pass

        return True


# ====== TÍNH TRỌNG SỐ LỚP THEO LEN-FREQ (không đụng dataloader/model) ======
def compute_class_weights_lenfreq(train_ds, alpha: float = 0.5):
    """
    Trọng số lớp ~ (1/freq_c) * (1/len_mean_c)^alpha, rồi chuẩn hoá mean=1.0
    - freq_c: số mẫu lớp c
    - len_mean_c: độ dài trung bình (số mẫu audio) lớp c
    alpha=0.5: vừa phải để xử lý bias độ dài mà không quá cực đoan
    """
    import numpy as np
    # thu thập (label, length)
    label_ids = []
    lengths = []
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

    inv_freq = (freq.mean() / freq)                  # mean-normalized inverse frequency
    inv_len  = (len_mean.mean() / len_mean) ** alpha # chống bias độ dài
    w = inv_freq * inv_len

    # chuẩn hoá mean=1.0 để không phá LR
    w = w * (C / w.sum())
    return torch.tensor(w, dtype=torch.float32)


# === Script entry: train, evaluate, and save HDF5 ===
if __name__ == "__main__":
    import json, h5py
    from pathlib import Path
    import numpy as np
    from loading.dataloader import build_train_test_dataset, VNEMOSDataset
    from training.callbacks import CheckpointsCallback
    from training.optimizers import split_param_groups, build_optimizer
    from transformers import get_cosine_schedule_with_warmup

    # 1) Config tối thiểu (giữ nguyên pipeline, KHÔNG dùng crop/window)
    cfg = Config(
        name="MER_CLI_run_lenfreq_maxpool",
        checkpoint_dir="checkpoints/mer_cli_run_lenfreq_maxpool",
        num_epochs=50,
        batch_size=8,
        num_workers=2,

        learning_rate=2e-5,
        optimizer_type="AdamW",
        scheduler_type="cosine_warmup",
        warmup_ratio=0.05,
        scheduler_step_unit="step",

        # Loss: CE có trọng số (thiết lập ở dưới)
        loss_type="CrossEntropyLoss",
        label_smoothing=0.0,

        # Data
        data_root="output",
        jsonl_dir="",
        sample_rate=16000,
        max_audio_sec=None,
        text_max_length=96,

        # Sampler/bucketing (nếu dataloader đã hỗ trợ)
        use_length_bucket=True,
        length_bucket_size=64,
        bucketing_text_alpha=0.03,
        use_weighted_sampler=True,
        lenfreq_alpha=0.5,

        # Model
        model_type="MemoCMT",
        text_encoder_type="phobert",
        text_encoder_ckpt="vinai/phobert-base",
        text_encoder_dim=768,
        text_unfreeze=False,   # sẽ mở nhẹ vài lớp cuối ở dưới (không đổi kiến trúc)
        audio_encoder_type="wav2vec2_xlsr",
        audio_encoder_ckpt="facebook/wav2vec2-large-xlsr-53",
        audio_encoder_dim=1024,
        audio_unfreeze=False,  # sẽ mở nhẹ vài lớp cuối ở dưới
        fusion_dim=768,
        fusion_head_output_type="max",   # ← dùng max pooling tạm thời, KHÔNG sửa kiến trúc
        linear_layer_output=[256, 128],
        dropout=0.10,

        # Runtime tricks
        use_amp=True,
        max_grad_norm=1.0,

        # Checkpoints
        save_best_val=True,
        max_to_keep=2,
    )
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 2) Data loaders + id2label
    train_loader, eval_loader = build_train_test_dataset(cfg)
    _train_ds = VNEMOSDataset(cfg, "train")
    label2id = _train_ds.label2id
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    cfg.num_classes = len(id2label)

    # 3) Model
    net = MER(cfg, device="cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu")
    net = net.to("cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu")

    # (Tuỳ chọn) Mở nhẹ **last-k layers** để tăng khả năng học cho lớp khó (không đổi kiến trúc)
    # Text (Roberta/PhoBERT): mở các layer 8..11
    for n, p in net.text_encoder.named_parameters():
        if ".encoder.layer." in n and any(f".{i}." in n for i in [8, 9, 10, 11]):
            p.requires_grad = True
    # Audio (Wav2Vec2): mở các layer 20..23
    for n, p in net.audio_encoder.named_parameters():
        if ".encoder.layers." in n and any(f".{i}." in n for i in [20, 21, 22, 23]):
            p.requires_grad = True

    # 4) Loss = CrossEntropy có trọng số theo len-freq
    class_weights = compute_class_weights_lenfreq(_train_ds, alpha=0.5).to(
        "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
    )
    print("Class weights (len-freq):", class_weights.tolist())
    criterion = CrossEntropyLoss(cfg, weight=class_weights)

    trainer = Trainer(cfg, net, criterion, log_dir="logs")

    # 5) Optimizer (tách LR encoder/head) + scheduler cosine warmup
    enc_lr, head_lr = 5e-6, 2e-4      # ↑ tăng head LR để đầu phân loại học quyết định tốt hơn
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

    # 6) Callback lưu best theo macro-F1
    from training.callbacks import CheckpointsCallback
    ckpt_cb = CheckpointsCallback(
        cfg.checkpoint_dir,
        save_freq=200,
        max_to_keep=cfg.max_to_keep,
        save_best_val=True,
        monitor="val_macro_f1",
        mode="max",
    )
    callbacks = [ckpt_cb]

    # 7) Train
    trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=eval_loader, callbacks=callbacks)
    best_path = getattr(ckpt_cb, "best_path", "")
    print("Best checkpoint:", best_path)

    # 8) Load best trước khi đánh giá & xuất h5 (nếu có)
    if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
        print("Loading best weights:", best_path)
        state = torch.load(best_path, map_location=trainer.device)
        trainer.network.load_state_dict(state)

    # 9) Đánh giá chi tiết (giống notebook)
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

    # 10) Lưu trọng số ra HDF5 (.h5) với TÊN MỚI tránh xung đột
    import h5py, json, datetime
    h5_name = "model_weights_lenfreq_maxpool.h5"  # tên mới cố định cho lần chạy này
    h5_path = Path(cfg.checkpoint_dir) / h5_name

    state = trainer.network.state_dict()
    # Chuẩn hoá attr string để tránh lỗi dtype h5py
    torch_ver = str(getattr(torch, "__version__", "unknown"))
    id2label_json = json.dumps([k for k, v in sorted(label2id.items(), key=lambda x: x[1])], ensure_ascii=False)

    with h5py.File(h5_path, "w") as f:
        # metadata
        f.attrs["model_name"] = "MER"
        f.attrs["num_classes"] = int(cfg.num_classes)
        f.attrs["id2label"] = id2label_json
        f.attrs["torch_version"] = torch_ver
        # cấu hình (serialize)
        cfg_group = f.create_group("config")
        for k, v in vars(cfg).items():
            try:
                cfg_group.attrs[k] = json.dumps(v, ensure_ascii=False)
            except TypeError:
                cfg_group.attrs[k] = str(v)
        # weights
        weights_group = f.create_group("state_dict")
        for name, tensor in state.items():
            arr = tensor.detach().cpu().numpy()
            dset = weights_group.create_dataset(name, data=arr)
            dset.attrs["shape"] = arr.shape
            dset.attrs["dtype"] = str(arr.dtype)
    print(f"✅ Saved HDF5 weights to: {h5_path.resolve()}")
