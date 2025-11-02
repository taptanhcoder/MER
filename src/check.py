# src/check.py
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
import numpy as np

USE_CUDA_ENV = os.getenv("USE_CUDA", "1").strip()
FORCE_CPU = (USE_CUDA_ENV == "0")
DEVICE = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")

from configs.base import Config
from loading.dataloader import VNEMOSDataset, build_train_test_dataset
from model.networks import MER
from transformers import BatchEncoding


def _to_device_text(x, device):
    if isinstance(x, BatchEncoding):
        x = x.to(device)
        return {k: v for k, v in x.items()}
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def find_default_weights(cfg: Config) -> Optional[Path]:

    ckpt_dir = Path(cfg.checkpoint_dir)
    best_dir = ckpt_dir / "best_val_macro_f1"
    if best_dir.exists():
        pths = sorted(best_dir.glob("*.pth"))
        if pths:
            return pths[-1]
    h5 = ckpt_dir / "model_weights_lenfreq_maxpool.h5"
    if h5.exists():
        return h5
    return None


def load_h5_state_dict(h5_path: Path) -> Dict[str, torch.Tensor]:
    import h5py
    state = {}
    with h5py.File(h5_path, "r") as f:
        if "state_dict" not in f:
            raise RuntimeError(f"H5 file không có group 'state_dict': {h5_path}")
        g = f["state_dict"]
        for name in g.keys():
            arr = np.array(g[name])
            state[name] = torch.from_numpy(arr)
    return state


@torch.no_grad()
def evaluate_and_collect_errors(
    net: MER,
    loader,
    label2id: Dict[str, int],
    items: List[dict],
    device: torch.device,
) -> Tuple[float, float, np.ndarray, List[dict]]:

    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    num_classes = len(id2label)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)


    by_uid = {it.get("utterance_id"): it for it in items}

    all_preds, all_tgts = [], []
    errors = []

    net.eval()
    for batch in loader:
        ((tok, audio, labels), meta) = batch 
        audio  = audio.to(device)
        labels = labels.to(device)
        tok    = _to_device_text(tok, device)

        logits, *_ = net(tok, audio, meta=meta)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)


        for t, p in zip(labels, preds):
            cm[t.long(), p.long()] += 1


        for i in range(labels.size(0)):
            t = int(labels[i].item())
            p = int(preds[i].item())
            if p != t:
                uid = meta.get("utterance_id", [None] * labels.size(0))[i]
                it = by_uid.get(uid, {})
                row = {
                    "utterance_id": uid,
                    "speaker_id": meta.get("speaker_id", [None] * labels.size(0))[i],
                    "wav_path": it.get("wav_path"),
                    "text": it.get("transcript"),
                    "true_label_id": t,
                    "true_label": id2label[t] if t < num_classes else str(t),
                    "pred_label_id": p,
                    "pred_label": id2label[p] if p < num_classes else str(p),
                    "confidence": float(probs[i, p].item()),
                    "duration_sec": float(it.get("end", 0.0)) - float(it.get("start", 0.0)),
                }
                errors.append(row)

        all_preds.append(preds.cpu())
        all_tgts.append(labels.cpu())

    all_preds = torch.cat(all_preds) if all_preds else torch.zeros(0, dtype=torch.long)
    all_tgts  = torch.cat(all_tgts)  if all_tgts  else torch.zeros(0, dtype=torch.long)

    overall_acc = (all_preds == all_tgts).float().mean().item() if all_preds.numel() > 0 else 0.0


    def macro_f1_from_cm(cm_: torch.Tensor) -> float:
        f1s = []
        for c in range(cm_.size(0)):
            tp = cm_[c, c].item()
            fp = (cm_[:, c].sum() - cm_[c, c]).item()
            fn = (cm_[c, :].sum() - cm_[c, c]).item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return float(sum(f1s) / len(f1s)) if f1s else 0.0

    macro_f1 = macro_f1_from_cm(cm)

    return overall_acc, macro_f1, cm.numpy(), errors


def save_errors(outdir: Path, errors: List[dict], id2label: List[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # JSONL
    jsonl_path = outdir / "misclassified.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    import csv
    csv_path = outdir / "misclassified.csv"
    fields = [
        "utterance_id", "speaker_id", "wav_path", "duration_sec",
        "true_label", "pred_label", "confidence", "text"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in errors:
            w.writerow({
                "utterance_id": e.get("utterance_id"),
                "speaker_id": e.get("speaker_id"),
                "wav_path": e.get("wav_path"),
                "duration_sec": f"{e.get('duration_sec', 0.0):.3f}",
                "true_label": e.get("true_label"),
                "pred_label": e.get("pred_label"),
                "confidence": f"{e.get('confidence', 0.0):.4f}",
                "text": e.get("text"),
            })
    print(f"⛳ Saved misclassified to:\n  - {jsonl_path}\n  - {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Check misclassified samples on test/valid split.")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Đường dẫn tới .pth hoặc .h5. Nếu bỏ trống sẽ auto-tìm theo checkpoint_dir.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="checkpoints/check_results",
        help="Thư mục ghi kết quả (JSONL/CSV).",
    )
    args = parser.parse_args()

    # ===== Config giống train (không đụng kiến trúc/dataloader) =====
    cfg = Config(
        name="MER_CLI_run_lenfreq_maxpool",
        checkpoint_dir="checkpoints/mer_cli_run_lenfreq_maxpool",
        data_root="output",
        jsonl_dir="",
        sample_rate=16000,
        max_audio_sec=None,
        text_max_length=96,
        # Quan trọng: cùng kiểu pooling để nhất quán với run gần đây
        fusion_head_output_type="max",
        # Model encoders (đúng checkpoint)
        text_encoder_type="phobert",
        text_encoder_ckpt="vinai/phobert-base",
        text_encoder_dim=768,
        audio_encoder_type="wav2vec2_xlsr",
        audio_encoder_ckpt="facebook/wav2vec2-large-xlsr-53",
        audio_encoder_dim=1024,
        fusion_dim=768,
        linear_layer_output=[256, 128],
        dropout=0.10,
        num_workers=2,
        batch_size=8,
    )

    # ===== Data (dùng split test nếu có, else valid) =====
    # Ta dựng loaders như train, rồi dùng eval_loader (test/valid)
    _, eval_loader = build_train_test_dataset(cfg)
    # Lấy toàn bộ item để join metadata qua utterance_id
    train_ds = VNEMOSDataset(cfg, "train")
    label2id = train_ds.label2id
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    cfg.num_classes = len(id2label)

    # Ta cũng muốn có items của split eval để bổ sung wav_path, text, duration
    # VNEMOSDataset tự detect 'valid' nếu tồn tại
    eval_split = "valid"
    base_dir = (Path(cfg.data_root) / (getattr(cfg, "jsonl_dir", "") or "")).resolve()
    if (base_dir / "test.jsonl").exists():
        eval_split = "test"
    eval_ds = VNEMOSDataset(cfg, eval_split, label2id=label2id)
    eval_items = eval_ds.items  # list of dict

    # ===== Model =====
    net = MER(cfg, device=DEVICE.type).to(DEVICE)

    # ===== Load weights =====
    weight_path = Path(args.weights) if args.weights else find_default_weights(cfg)
    if weight_path is None:
        raise FileNotFoundError(
            "Không tìm thấy weights. Hãy chỉ định --weights PATH, hoặc kiểm tra checkpoint_dir."
        )
    print(f"Loading weights: {weight_path}")

    if weight_path.suffix.lower() == ".pth":
        state = torch.load(weight_path, map_location=DEVICE)
        net.load_state_dict(state, strict=True)
    elif weight_path.suffix.lower() == ".h5":
        state = load_h5_state_dict(weight_path)
        net.load_state_dict(state, strict=True)
    else:
        raise ValueError("Unsupported weight file. Chỉ hỗ trợ .pth hoặc .h5")

    # ===== Evaluate & collect errors =====
    acc, mf1, cm, errors = evaluate_and_collect_errors(net, eval_loader, label2id, eval_items, DEVICE)

    print("=== Evaluation (check.py) ===")
    print(f"Overall Acc : {acc:.4f}")
    print(f"Macro F1    : {mf1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # ===== Save errors to outdir =====
    outdir = Path(args.outdir)
    save_errors(outdir, errors, id2label)


if __name__ == "__main__":
    main()
