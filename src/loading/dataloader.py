# src/loading/dataloader.py
import os, json, re, pickle
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import unicodedata 


try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None
    _HAS_TORCHAUDIO = False
    import librosa

from configs.base import Config


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s is not None else ""

def _clean_text(s: str) -> str:
    s = _nfc(s)
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "NULL"

def _safe_resample(wave: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wave
    if _HAS_TORCHAUDIO:
        return torchaudio.functional.resample(wave, orig_sr, target_sr)
    w = wave.detach().cpu().numpy().astype(np.float32, copy=False)
    w_rs = librosa.resample(y=w, orig_sr=orig_sr, target_sr=target_sr)
    return torch.from_numpy(w_rs.astype(np.float32))


# ---------- VNEMOS ----------
class VNEMOSDataset(Dataset):

    def __init__(self, cfg: Config, split: str, label2id: Optional[Dict[str, int]] = None):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.data_root)
        jdir = getattr(cfg, "jsonl_dir", "")
        self.base_dir = ((root / jdir) if jdir else root).resolve()


        ar = getattr(cfg, "audio_root", None)
        self.audio_root = (Path(ar).resolve() if ar else self.base_dir.parent.resolve())

        jsonl = self.base_dir / f"{split}.jsonl"
        if not jsonl.exists():
            raise FileNotFoundError(
                f"Không tìm thấy JSONL: {jsonl}\n"
                f"- Nếu notebook ở project root: cfg.data_root='output', cfg.jsonl_dir=''\n"
                f"- Nếu notebook ở 'src':        cfg.data_root='../output', cfg.jsonl_dir=''"
            )

        self.items: List[dict] = []
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        if label2id is None:
            labels = sorted({it["emotion"] for it in self.items})
            self.label2id = {lb: i for i, lb in enumerate(labels)}
        else:
            self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.sample_rate = int(getattr(cfg, "sample_rate", 16000))
        self.max_audio_sec = getattr(cfg, "max_audio_sec", None)

        self.durations = [float(it["end"]) - float(it["start"]) for it in self.items]

        self.tokenizer = AutoTokenizer.from_pretrained(
            getattr(cfg, "text_encoder_ckpt", "vinai/phobert-base"),
            use_fast=True
        )

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_wav(self, wav_path_str: str) -> Path:
        wav_path_str = wav_path_str.strip().replace("\\", "/")
        p = Path(wav_path_str)
        if p.is_absolute():
            return p

        cand1 = (self.base_dir / wav_path_str).resolve()
        cand2 = (self.audio_root / wav_path_str).resolve()

        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2

        name = Path(wav_path_str).name
        hits = list(self.audio_root.rglob(name))
        if hits:
            return hits[0]
        hits2 = list(self.base_dir.rglob(name))
        if hits2:
            return hits2[0]

        raise FileNotFoundError(f"Không tìm thấy WAV: {cand1} hoặc {cand2}")

    def __getitem__(self, idx: int) -> Dict:
        ex = self.items[idx]

        wav_file = self._resolve_wav(ex["wav_path"])
        wave, sr = sf.read(str(wav_file), dtype="float32", always_2d=False)
        if wave.ndim == 2:
            wave = wave.mean(axis=1)

        # crop theo start/end
        start = float(ex.get("start", 0.0))
        end   = float(ex.get("end", 0.0))
        if end and end > 0.0:
            s = int(max(0.0, start) * sr)
            e = min(int(end * sr), len(wave))
            wave = wave[s:e]

        wav_t = torch.from_numpy(wave.astype(np.float32, copy=False))
        wav_t = _safe_resample(wav_t, sr, self.sample_rate)

        if self.max_audio_sec is not None:
            max_len = int(round(self.max_audio_sec * self.sample_rate))
            if wav_t.numel() < max_len:
                wav_t = torch.nn.functional.pad(wav_t, (0, max_len - wav_t.numel()))
            else:
                wav_t = wav_t[:max_len]

        text = _clean_text(ex.get("transcript", ""))
        label = int(self.label2id[ex["emotion"]])

        return {
            "utterance_id": ex.get("utterance_id"),
            "speaker_id":   ex.get("speaker_id"),
            "text":         text,
            "audio":        wav_t,  
            "label":        label,
        }


class PickleDataset(Dataset):
    def __init__(self, cfg: Config, data_mode: str = "train.pkl"):
        super().__init__()
        p = Path(cfg.data_root) / data_mode
        if not p.exists():
            raise FileNotFoundError(f"Không thấy PKL: {p} (code này mặc định dùng JSONL).")
        with p.open("rb") as f:
            self.data_list = pickle.load(f)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        audio_path, text, label = self.data_list[idx]
        wave, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        wav_t = torch.from_numpy(wave.astype(np.float32))
        wav_t = _safe_resample(wav_t, sr, getattr(self.cfg, "sample_rate", 16000))
        return {"text": _clean_text(text), "audio": wav_t, "label": int(label)}



class Collator:
    def __init__(self, tokenizer: AutoTokenizer, text_max_length: int = 64):
        self.tok = tokenizer
        self.text_max_length = text_max_length

    def __call__(self, batch: List[Dict]):

        audios = [b["audio"] for b in batch]
        lengths = [a.numel() for a in audios]  
        T = max(a.numel() for a in audios)
        aud = [a if a.numel() == T else torch.nn.functional.pad(a, (0, T - a.numel())) for a in audios]
        audio_tensor = torch.stack(aud, dim=0) 

        texts = [b["text"] for b in batch]
        tok = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
            add_prefix_space=True,
            pad_to_multiple_of=8,  
        )

        tok = {k: v for k, v in tok.items()}

        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        out = (tok, audio_tensor, labels)
        if "utterance_id" in batch[0]:
            meta = {
                "utterance_id": [b.get("utterance_id") for b in batch],
                "speaker_id":   [b.get("speaker_id") for b in batch],
                "audio_lengths": lengths,  
            }
            return out, meta
        return out



import math, random
from torch.utils.data import Sampler

class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size=4, shuffle=True, bucket_size=64):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.bucket_size = int(bucket_size)

        idxs = list(range(len(lengths)))
        if self.shuffle:
            random.shuffle(idxs)
        chunks = [idxs[i:i+self.bucket_size] for i in range(0, len(idxs), self.bucket_size)]
        chunks = [sorted(ch, key=lambda i: lengths[i]) for ch in chunks]
        self.order = [i for ch in chunks for i in ch]

    def __iter__(self):
        bs = self.batch_size
        for i in range(0, len(self.order), bs):
            yield self.order[i:i+bs]

    def __len__(self):
        return math.ceil(len(self.order) / self.batch_size)


# ---------- Builder (JSONL only) ----------
def build_train_test_dataset(cfg: Config, encoder_model: Optional[object] = None):

    base_dir = (Path(cfg.data_root) / (getattr(cfg, "jsonl_dir", "") or "")).resolve()
    train_jsonl = base_dir / "train.jsonl"
    valid_jsonl = base_dir / "valid.jsonl"
    test_jsonl  = base_dir / "test.jsonl"

    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"Không thấy {train_jsonl}.\n"
            f"- Notebook ở project root → cfg.data_root='output', cfg.jsonl_dir=''\n"
            f"- Notebook ở 'src'        → cfg.data_root='../output', cfg.jsonl_dir=''"
        )

    train_set = VNEMOSDataset(cfg, "train")
    lbl2id = train_set.label2id
    eval_split = "valid" if valid_jsonl.exists() else "test"
    eval_set = VNEMOSDataset(cfg, eval_split, lbl2id)

    collate = Collator(train_set.tokenizer, text_max_length=getattr(cfg, "text_max_length", 64))

    pin = torch.cuda.is_available()
    nw = max(0, int(getattr(cfg, "num_workers", 0)))
    persistent = True if nw > 0 else False

    use_len_bucket = bool(getattr(cfg, "use_length_bucket", False))
    bucket_size = int(getattr(cfg, "length_bucket_size", 64))

    if use_len_bucket:
        train_sampler = LengthBucketBatchSampler(
            lengths=train_set.durations,
            batch_size=cfg.batch_size,
            shuffle=True,
            bucket_size=bucket_size,
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate,
            pin_memory=pin,
            persistent_workers=persistent,
        )

    eval_loader = DataLoader(
        eval_set,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return train_loader, eval_loader
