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
            if value.lstrip("-").replace(".", "", 1).isdigit():
                if "." in value:
                    return float(value)
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
            key, value = line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip()
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
        self.learning_rate: float = 3e-5
        self.learning_rate_step_size: int = 30
        self.learning_rate_gamma: float = 0.1
        self.optimizer_type: str = "AdamW"
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.adam_eps = 1e-08
        self.adam_weight_decay = 0.01
        self.momentum = 0.99
        self.sgd_weight_decay = 1e-6

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

        # Dataset
        self.data_name: str = "VNEMOS"
        self.data_root: str = "output"
        self.jsonl_dir: str = ""
        self.data_valid: Union[str, None] = None
        self.sample_rate: int = 16000
        self.max_audio_sec: float = None
        self.text_max_length: int = 96

        # Samplers
        self.use_length_bucket: bool = True
        self.length_bucket_size: int = 64
        self.bucketing_text_alpha: float = 0.03
        self.use_weighted_sampler: bool = True
        self.lenfreq_alpha: float = 0.5

        # Model 
        self.num_classes: int = 5
        self.num_attention_head: int = 8
        self.dropout: float = 0.10
        self.model_type: str = "MemoCMT"

        # Text encoder 
        self.text_encoder_type: str = "phobert"          
        self.text_encoder_ckpt: str = "vinai/phobert-base"
        self.text_encoder_dim: int = 768
        self.text_unfreeze: bool = False

        # Audio encoder 
        self.audio_encoder_type: str = "wav2vec2_xlsr"
        self.audio_encoder_ckpt: str = "facebook/wav2vec2-large-xlsr-53"  
        self.audio_encoder_dim: int = 1024
        self.audio_unfreeze: bool = False

        # Whisper feature extractor override (None -> mặc định)
        self.whisper_n_fft: Union[int, None] = None
        self.whisper_hop_length: Union[int, None] = None
        self.whisper_win_length: Union[int, None] = None
        self.whisper_nb_mels: Union[int, None] = None

        # W2V2 + VGGish merge mode: "concat" | "sum"
        self.w2v2_vggish_merge: str = "concat"

        # Fourier2Vec hyper
        self.fourier_n_mels: int = 64
        self.fourier_fmin: float = 125.0
        self.fourier_fmax: float = 7500.0
        self.fourier_win_ms: float = 25.0
        self.fourier_hop_ms: float = 10.0
        self.fourier_patch_len: int = 1
        self.fourier_patch_hop: int = 1
        self.fourier_hidden_size: int = 256
        self.fourier_num_heads: int = 4
        self.fourier_num_layers: int = 4

        # Fusion chung
        self.fusion_dim: int = 768
        self.fusion_head_output_type: str = "cls"  
        self.linear_layer_output: List = [256, 128]

        self.fusion_type: str = "xattn"

        self.fusion_bilstm_hidden_text: int = self.fusion_dim // 2
        self.fusion_bilstm_hidden_audio: int = self.fusion_dim // 2
        self.fusion_bilstm_layers: int = 1
        self.fusion_bilstm_dropout: float = 0.1
        self.fusion_bilstm_bidirectional: bool = True
        self.fusion_blocks: int = 1
        self.fusion_merge: str = "concat"        
        self.fusion_pool_heads: int = 1          


        self.fusion_cnn_layers: int = 2
        self.fusion_cnn_kernel: int = 3
        self.fusion_cnn_dropout: float = 0.10
        self.fusion_cnn_dilation_growth: int = 1


        self.use_amp: bool = True
        self.max_grad_norm: float = 1.0

        for key, value in kwargs.items():
            setattr(self, key, value)
