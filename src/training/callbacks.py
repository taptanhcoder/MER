import logging
import os
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

        # Train phase: lưu định kỳ theo step
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

        # Val phase: lưu best theo metric
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
