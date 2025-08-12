import logging
import os
from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from configs.base import Config
from model.networks import MemoCMT
from training.trainer import TorchTrainer



def _to_device_text(x, device):

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

    def _unpack_batch(self, batch: Union[Tuple, Tuple[Tuple, Dict]]):
        """
        batch có thể là:
          (input_text, input_audio, label)
        hoặc:
          ((input_text, input_audio, label), meta)
        """
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            data, _meta = batch
            return data
        return batch

    def train_step(self, batch) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)


        input_text, input_audio, label = self._unpack_batch(batch)
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = _to_device_text(input_text, self.device)


        if self.use_amp:
            with torch.cuda.amp.autocast():

                logits, *_ = self.network(input_text, input_audio)
                loss = self.criterion(logits, label)

            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, *_ = self.network(input_text, input_audio)
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

        input_text, input_audio, label = self._unpack_batch(batch)
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = _to_device_text(input_text, self.device)

        with torch.no_grad():
            logits, *_ = self.network(input_text, input_audio)
            loss = self.criterion(logits, label)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).float().mean()


        return {
            "val_loss": float(loss.detach().cpu()),
            "val_acc": float(acc.detach().cpu()),
        }
