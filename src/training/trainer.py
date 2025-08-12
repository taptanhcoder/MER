import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import tqdm

from . import optimizers
from .callbacks import Callback

# mlflow optional
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TorchTrainer(ABC, nn.Module):
    def __init__(self, log_dir: str = "logs"):
        super().__init__()
        self.log_dir = log_dir
        self.global_step = 0
        self.start_epoch = 1
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    # ----- inference -----
    def predict(self, inputs: Union[torch.Tensor, Dict, List]):
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)

    # ----- one epoch -----
    def train_epoch(
        self,
        step: int,
        epoch: int,
        train_data: Iterable,
        eval_data: Optional[Iterable] = None,
        logger: Optional[logging.Logger] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.network.train()
        if logger is None:
            logger = logging.getLogger("Training")

        epoch_log: Dict[str, List[float]] = {}
        with tqdm.tqdm(total=len(train_data), ascii=True, desc=f"Epoch {epoch}") as pbar:
            for batch in train_data:
                step += 1
                train_log = self.train_step(batch)
                assert isinstance(train_log, dict), "train_step phải trả dict các scalar."


                postfix = []
                for k, v in train_log.items():
                    epoch_log.setdefault(k, []).append(float(v))
                    postfix.append(f"{k}: {float(v):.4f}")
                    if _HAS_MLFLOW:
                        try:
                            mlflow.log_metric(f"train_{k}", float(v), step=step)
                        except Exception:
                            pass
                pbar.set_postfix_str(" | ".join(postfix))
                pbar.update(1)


                try:
                    if self.optimizer is not None:
                        lr = self.optimizer.param_groups[0]["lr"]
                        if _HAS_MLFLOW:
                            mlflow.log_metric("learning_rate", lr, step=step)
                except Exception:
                    pass

                # callbacks train
                if callbacks:
                    for cb in callbacks:
                        cb(self, step, epoch, train_log, isValPhase=False, logger=logger)

        for k, vals in epoch_log.items():
            m = float(np.mean(vals))
            logger.info(f"Epoch {epoch} - {k}: {m:.4f}")
            if _HAS_MLFLOW:
                try:
                    mlflow.log_metric(f"train_epoch_{k}", m, step=step)
                except Exception:
                    pass

        # validation
        if eval_data is not None:
            self.network.eval()
            logger.info("Performing validation...")
            eval_logs: Dict[str, List[float]] = {}

            for batch in tqdm.tqdm(eval_data, ascii=True, desc="Valid"):
                val_log = self.test_step(batch)
                assert isinstance(val_log, dict), "test_step phải trả dict các scalar."
                for k, v in val_log.items():
                    eval_logs.setdefault(k, []).append(float(v))

            agg = {k: float(np.mean(vs)) for k, vs in eval_logs.items()}
            postfix = " ".join([f"{k}: {val:.4f}" for k, val in agg.items()])
            logger.info("Validation: " + postfix)
            if _HAS_MLFLOW:
                for k, val in agg.items():
                    try:
                        mlflow.log_metric(f"val_{k}", val, step=step)
                    except Exception:
                        pass

            if callbacks:
                for cb in callbacks:
                    cb(self, step, epoch, agg, isValPhase=True, logger=logger)

        return step

    # ----- test -----
    def evaluate(self, test_data: Iterable, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
        self.network.eval()
        if logger is None:
            logger = logging.getLogger("Training")
        test_logs: Dict[str, List[float]] = {}

        for batch in tqdm.tqdm(test_data, ascii=True, desc="Test"):
            out = self.test_step(batch)
            for k, v in out.items():
                test_logs.setdefault(k, []).append(float(v))

        agg = {k: float(np.mean(vs)) for k, vs in test_logs.items()}
        postfix = " ".join([f"{k}: {val:.4f}" for k, val in agg.items()])
        logger.info("Test: " + postfix)
        if _HAS_MLFLOW:
            for k, val in agg.items():
                try:
                    mlflow.log_metric(f"test_{k}", val, step=self.global_step)
                except Exception:
                    pass
        return agg

    # ----- save/load -----
    def save(self, path: str, step=None) -> str:
        os.makedirs(path, exist_ok=True)
        ckpt_path = os.path.join(path, f"checkpoint_{step}.pt")
        torch.save(self.network, ckpt_path)
        return ckpt_path

    @classmethod
    def load(cls, path: str):
        return torch.load(path)

    def save_weights(self, path: str, step=None):
        os.makedirs(path, exist_ok=True)
        ckpt_path = os.path.join(path, f"checkpoint_{step}.pth")
        torch.save(self.network.state_dict(), ckpt_path)
        return ckpt_path

    def load_weights(self, path: str, device: str = "cpu"):
        state = torch.load(path, map_location=device)
        self.network.load_state_dict(state)

    def save_all_states(self, path: str, global_epoch: int, global_step: int):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "epoch": global_epoch,
            "global_step": global_step,
            "state_dict_network": self.network.state_dict(),
            "state_optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "state_lr_scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        ckpt_path = os.path.join(path, f"checkpoint_{global_epoch}_{global_step}.pt")
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load_all_states(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device)
        self.start_epoch = int(ckpt.get("epoch", 1))
        self.global_step = int(ckpt.get("global_step", 0))
        self.network.load_state_dict(ckpt["state_dict_network"])
        if self.optimizer and ckpt.get("state_optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["state_optimizer"])
        if self.scheduler and ckpt.get("state_lr_scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["state_lr_scheduler"])
        logging.info(f"Loaded checkpoint from {path}. Resume at epoch {self.start_epoch}")

    # ----- compile/fit -----
    def compile(
        self,
        optimizer: Union[str, torch.optim.Optimizer] = "adamw",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        *,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.9,
        param_groups: Union[None, str, List[dict]] = "auto",
    ):
        if not hasattr(self, "network"):
            raise AttributeError("Hãy gán self.network trong __init__ trainer của bạn.")

        if isinstance(optimizer, str):
            if param_groups is None:
                params = self.network.parameters()
            elif param_groups == "auto":
                params = optimizers.split_param_groups(self, lr_enc=lr * 0.25, lr_head=lr, weight_decay=weight_decay)
            elif isinstance(param_groups, list):
                params = param_groups
            else:
                raise ValueError("param_groups must be None | 'auto' | List[dict]")

            self.optimizer = optimizers.build_optimizer(
                optimizer, params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, momentum=momentum
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

    def fit(
        self,
        train_data: Iterable,
        epochs: int,
        eval_data: Optional[Iterable] = None,
        test_data: Optional[Iterable] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        if self.optimizer is None:
            raise AttributeError("Please compile the model first!")
        assert (isinstance(callbacks, list) or callbacks is None), "Callbacks must be a list"

        run_dir = os.path.join(self.log_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        logging.getLogger().setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger = logging.getLogger("Training")
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())

        if _HAS_MLFLOW:
            mlflow.set_tracking_uri(uri=f'file://{os.path.abspath(os.path.join(run_dir, "mlruns"))}')

        global_step = self.global_step

        def _one_run():
            nonlocal global_step
            for epoch in range(self.start_epoch, epochs + 1):
                logger.info(f"Epoch {epoch}/{epochs}")
                global_step = self.train_epoch(global_step, epoch, train_data, eval_data, logger, callbacks)
                self.lr_scheduler(global_step, epoch)
                if test_data is not None:
                    self.evaluate(test_data, logger=logger)

        if _HAS_MLFLOW:
            with mlflow.start_run():
                _one_run()
        else:
            _one_run()

    def lr_scheduler(self, step: int, epoch: int) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    # ----- abstract steps -----
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Thực hiện 1 bước train (loss.backward(), optimizer.step(), zero_grad) và trả dict scalar."""
        ...

    @abstractmethod
    def test_step(self, batch: Any) -> Dict[str, float]:
        """Thực hiện 1 bước eval và trả dict scalar."""
        ...
