import logging
import os
from abc import ABC, abstractmethod
from typing import List, Union


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        ...

    @abstractmethod
    def save(self, cfg):
        ...


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, cfg):
        message = "\n"
        for k, v in sorted(vars(cfg).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(cfg.checkpoint_dir), exist_ok=True)
        out_opt = os.path.join(cfg.checkpoint_dir, "cfg.log")
        with open(out_opt, "w") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logging.info(message)

    def load(self, cfg_path: str):
        def decode_value(value: str):
            value = value.strip()
            if "." in value and value.replace(".", "").isdigit():
                return float(value)
            if value.isdigit():
                return int(value)
            if value == "True":
                return True
            if value == "False":
                return False
            if value == "None":
                return None
            if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                return value[1:-1]
            return value

        with open(cfg_path, "r") as f:
            data = [d for d in f.read().split("\n") if d]
        data_dict = {}
        for line in data:
            key, value = line.split(":")[0].strip(), line.split(":")[1].strip()
            if value.startswith("[") and value.endswith("]"):
                value = [decode_value(x) for x in value[1:-1].split(",")]
            else:
                value = decode_value(value)
            data_dict[key] = value
        for k, v in data_dict.items():
            setattr(self, k, v)


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # Training
        self.trainer: str = "Trainer"
        self.num_epochs: int = 20
        self.checkpoint_dir: str = "checkpoints"
        self.save_all_states: bool = False
        self.save_best_val: bool = True
        self.max_to_keep: int = 2
        self.save_freq: int = 1000
        self.batch_size: int = 8
        self.num_workers: int = 2

        # LR/Optim
        self.learning_rate: float = 3e-5           # head LR đề xuất
        self.learning_rate_step_size: int = 30
        self.learning_rate_gamma: float = 0.1
        self.optimizer_type: str = "AdamW"
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.adam_eps = 1e-08
        self.adam_weight_decay = 0.01
        self.momemtum = 0.99
        self.sdg_weight_decay = 1e-6

        # Scheduler
        self.scheduler_type: str = "cosine_warmup" 
        self.warmup_ratio: float = 0.1           
        self.scheduler_step_unit: str = "step"     

        # Gradual unfreeze
        self.gradual_unfreeze_epoch: int = 1
        self.text_unfreeze_last_k: int = 4
        self.audio_unfreeze_last_k: int = 4

        # Resume
        self.resume: bool = False
        self.resume_path: Union[str, None] = None
        self.cfg_path: Union[str, None] = None

        # Loss
        self.loss_type: str = "FocalLoss"   
        self.loss_gamma: float = 1.5         
        self.label_smoothing: float = 0.05

        # Dataset (VNEMOS JSONL)
        self.data_name: str = "VNEMOS"
        self.data_root: str = "output"
        self.jsonl_dir: str = ""
        self.data_valid: Union[str, None] = None
        self.sample_rate: int = 16000
        self.max_audio_sec: float = None     
        self.text_max_length: int = 96       

        # Length-bucket sampler
        self.use_length_bucket: bool = True
        self.length_bucket_size: int = 64

        # Model
        self.num_classes: int = 4
        self.num_attention_head: int = 8
        self.dropout: float = 0.10
        self.model_type: str = "MemoCMT"

        # Text: PhoBERT
        self.text_encoder_type: str = "phobert"
        self.text_encoder_ckpt: str = "vinai/phobert-base"
        self.text_encoder_dim: int = 768
        self.text_unfreeze: bool = False

        # Audio: Wav2Vec2 XLSR-53
        self.audio_encoder_type: str = "wav2vec2_xlsr"
        self.audio_encoder_ckpt: str = "facebook/wav2vec2-large-xlsr-53"
        self.audio_encoder_dim: int = 1024
        self.audio_unfreeze: bool = False

        # Fusion
        self.fusion_dim: int = 768
        self.fusion_head_output_type: str = "cls" 
        self.linear_layer_output: List = [256, 128]

        # Train tricks
        self.use_amp: bool = True
        self.max_grad_norm: float = 1.0

        for key, value in kwargs.items():
            setattr(self, key, value)
