# src/training/trainer.py
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import BatchEncoding

from configs.base import Config
from model.networks import MemoCMT, MER


# =========================
# Base trainer (no dependency)
# =========================
class TorchTrainer:

    def __init__(self, log_dir: str = "logs"):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = None
        self.scheduler = None
        self.scheduler_step_unit = "step"
        self.global_step = 0
        self.global_epoch = 0

    # ---- lifecycle ----
    def compile(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        lr: Optional[float] = None,
        param_groups: Optional[object] = None,
        scheduler_step_unit: str = "step",
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_unit = scheduler_step_unit

    # ---- required-by-subclass ----
    def train_step(self, batch) -> Dict[str, float]:
        raise NotImplementedError

    def test_step(self, batch) -> Dict[str, float]:
        raise NotImplementedError

    # ---- checkpoint I/O ----
    def save_weights(self, dirpath: str, tag: int) -> str:

        out = Path(dirpath) / (f"weights_step_{tag}.pth" if tag else "weights_best.pth")
        state = self.network.state_dict()
        torch.save(state, out)
        return str(out)

    def save_all_states(self, dirpath: str, epoch: int, tag: int) -> str:

        out = Path(dirpath) / (f"state_e{epoch}_s{tag}.pth" if tag else "state_best.pth")
        state = {
            "model": self.network.state_dict(),
            "optimizer": getattr(self, "optimizer", None).state_dict() if self.optimizer else None,
            "scheduler": getattr(self, "scheduler", None).state_dict() if self.scheduler else None,
            "scaler": getattr(self, "scaler", None).state_dict() if hasattr(self, "scaler") else None,
            "epoch": epoch,
            "step": self.global_step,
        }
        torch.save(state, out)
        return str(out)

    # ---- helpers ----
    @staticmethod
    def _combine_logs_mean(logs_list: List[Dict[str, float]]) -> Dict[str, float]:
        agg = {}
        if not logs_list:
            return agg
        keys = logs_list[0].keys()
        for k in keys:
            vals = [float(x[k]) for x in logs_list if k in x]
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

    # ---- training loop ----
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
                step_logs = self.train_step(batch)  
                self.global_step += 1
                epoch_logs.append(step_logs)

  
                if self.scheduler is not None and self.scheduler_step_unit == "step":
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass


                logs_for_cb = dict(step_logs)
                for cb in callbacks:
                    try:
                        cb(self, self.global_step, self.global_epoch, logs_for_cb, isValPhase=False, logger=self.logger)
                    except Exception as e:
                        self.logger.warning("Callback error (train step): %s", e)

            # Aggregate epoch logs
            mean_logs = self._combine_logs_mean(epoch_logs)
            self.logger.info(f"[Epoch {epoch}] train: {mean_logs}")

            # Validation
            val_logs = {}
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

                val_logs = self._combine_logs_mean(vlogs)
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

                # Callbacks per validation phase
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



def _to_device_text(x, device):
    if isinstance(x, BatchEncoding):
        x = x.to(device)
        return {k: v for k, v in x.items()}
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: MemoCMT,
        criterion: torch.nn.Module = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.use_amp = bool(getattr(cfg, "use_amp", torch.cuda.is_available()))
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 0.0))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _standardize_batch(self, batch):
        # batch = ((tok, audio, labels), meta)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            a, meta = batch
            if isinstance(a, (tuple, list)) and len(a) == 3:
                tok, audio, labels = a
                return tok, audio, labels, meta
        raise ValueError(f"Batch structure không hỗ trợ: preview={str(batch)[:200]}")

    def _compute_loss(self, logits, labels, meta: Dict):
        if hasattr(self.criterion, "forward"):
            try:
                loss_name = self.criterion.__class__.__name__.lower()
                if "sampleweighted" in loss_name or "sample_weighted" in loss_name:
                    sw = meta.get("sample_weight", None)
                    return self.criterion(logits, labels, sample_weight=sw)
            except Exception:
                pass
        if labels.dtype != torch.long:
            labels = labels.long()
        return self.criterion(logits, labels)

    def train_step(self, batch) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        tok, audio, labels, meta = self._standardize_batch(batch)
        audio  = audio.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        tok    = _to_device_text(tok, self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                out = self.network(tok, audio, meta=meta)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = self._compute_loss(logits, labels, meta)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out = self.network(tok, audio, meta=meta)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = self._compute_loss(logits, labels, meta)
            loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        return {
            "loss": float(loss.detach().cpu()),
            "acc": float(acc.detach().cpu()),
        }

    def test_step(self, batch) -> Dict[str, float]:
        self.network.eval()

        tok, audio, labels, meta = self._standardize_batch(batch)
        audio  = audio.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        tok    = _to_device_text(tok, self.device)

        with torch.no_grad():
            out = self.network(tok, audio, meta=meta)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = self._compute_loss(logits, labels, meta)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        return {
            "val_loss": float(loss.detach().cpu()),
            "val_acc": float(acc.detach().cpu()),
            "preds": preds.detach().cpu(),
            "targets": labels.detach().cpu(),  
        }
