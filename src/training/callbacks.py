import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional


class Callback(ABC):
    @abstractmethod
    def __call__(
        self,
        trainer,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        ...


class CheckpointsCallback(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 1000,
        max_to_keep: int = 3,
        save_best_val: bool = False,
        save_all_states: bool = False,
        monitor: str = "val_loss",
        mode: str = "min",  # 'min' cho loss, 'max' cho acc/f1
    ):
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.save_freq = int(save_freq)
        self.max_to_keep = int(max_to_keep)
        self._kept_paths = []

        self.save_all_states = bool(save_all_states)

        self.save_best_val = bool(save_best_val)
        self.monitor = str(monitor)
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_path = ""

        if self.save_best_val:
            logging.warning(
                "save_best_val=True → cần truyền validation loader vào trainer.fit(). "
                "Best theo metric '%s' (%s).", self.monitor, self.mode
            )

    def _save(self, trainer, dirpath: Path, tag: int):
        dirpath.mkdir(parents=True, exist_ok=True)
        if self.save_all_states:
            return trainer.save_all_states(str(dirpath), 0, tag)
        else:
            return trainer.save_weights(str(dirpath), tag)

    def __call__(
        self,
        trainer,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        log = logger or logging.getLogger(__name__)


        if not isValPhase and self.save_freq > 0:
            if global_step % self.save_freq == 0:
                log.info("Saving checkpoint at step %d", global_step)
                ckpt_path = self._save(trainer, self.ckpt_dir, global_step)
                self._kept_paths.append(ckpt_path)
                while len(self._kept_paths) > self.max_to_keep:
                    old = self._kept_paths.pop(0)
                    try:
                        os.remove(old)
                        log.info("Deleted old checkpoint: %s", old)
                    except Exception as e:
                        log.warning("Failed to delete %s: %s", old, e)


        elif isValPhase and self.save_best_val:
            if self.monitor not in logs:
                log.warning("Monitor '%s' không có trong logs: %s", self.monitor, list(logs.keys()))
                return
            value = float(logs[self.monitor])
            improved = value < self.best_value if self.mode == "min" else value > self.best_value
            if improved:
                prev = self.best_value
                self.best_value = value
                best_dir = self.ckpt_dir / f"best_{self.monitor}"
                log.info("Improved %s from %.6f to %.6f. Saving best...", self.monitor, prev, value)
                ckpt_path = self._save(trainer, best_dir, 0)
                self.best_path = ckpt_path


class GradualUnfreezeCallback(Callback):

    def __init__(self, epoch_trigger: int = 1, text_last_k: int = 4, audio_last_k: int = 4):
        self.epoch_trigger = int(epoch_trigger)
        self.text_last_k = int(text_last_k)
        self.audio_last_k = int(audio_last_k)
        self.done = False

    def _set_requires_grad(self, module, pattern: str, last_k: int):
        layer_ids = set()
        for n, _ in module.named_parameters():
            m = re.search(pattern, n)
            if m:
                layer_ids.add(int(m.group(1)))
        if not layer_ids:
            return
        cutoff = set(sorted(layer_ids)[-last_k:])
        for n, p in module.named_parameters():
            m = re.search(pattern, n)
            if m and int(m.group(1)) in cutoff:
                p.requires_grad = True

    def __call__(self, trainer, global_step, global_epoch, logs, isValPhase=False, logger: Optional[logging.Logger] = None):
        if self.done or isValPhase or global_epoch < self.epoch_trigger:
            return
        log = logger or logging.getLogger(__name__)
        net = trainer.network
        # PhoBERT: encoder.layer.<i>.
        self._set_requires_grad(net.text_encoder, r"encoder\.layer\.(\d+)\.", self.text_last_k)
        # Wav2Vec2: encoder.layers.<i>.
        self._set_requires_grad(net.audio_encoder.model, r"encoder\.layers\.(\d+)\.", self.audio_last_k)
        self.done = True
        log.info(f"Gradual unfreeze done at epoch {global_epoch}: text_last_k={self.text_last_k}, audio_last_k={self.audio_last_k}")
