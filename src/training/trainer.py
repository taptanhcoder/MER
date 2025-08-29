import logging
import os
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from configs.base import Config
from model.networks import MemoCMT, MER
from training.trainer import TorchTrainer
from transformers import BatchEncoding


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
        }
