
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil
import csv
import sys
import unicodedata
import re

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s

def simplify_for_match(s: str) -> str:

    if not isinstance(s, str):
        return ""
    s0 = nfc(s).lower()
    s0 = s0.replace("copy of ", "copy_of_")
    s0 = s0.replace(" ", "_")
    s0 = re.sub(r"[^a-z0-9._-]+", "", s0)
    s0 = re.sub(r"_+", "_", s0)
    s0 = s0[:-4] if s0.endswith(".wav") else s0
    return s0

def ensure_wav_name(name: str) -> str:
    name = nfc(name)
    if not name.lower().endswith(".wav"):
        name = f"{name}.wav"
    return name

def read_jsonl(jsonl_path: Path) -> List[Dict]:
    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Bỏ qua dòng JSONL lỗi: {e}")
    return items

def list_all_wavs(root: Path) -> List[Path]:
    return list(root.rglob("*.wav"))

def build_wav_index(wav_files: List[Path]) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:

    exact_by_name: Dict[str, Path] = {}
    fuzzy_by_key: Dict[str, List[Path]] = {}
    for p in wav_files:
        base = p.name
        if base not in exact_by_name:
            exact_by_name[base] = p
        key = simplify_for_match(base)
        fuzzy_by_key.setdefault(key, []).append(p)
    return exact_by_name, fuzzy_by_key

def generate_name_candidates(rec: Dict) -> List[str]:

    cands = []
    wp = rec.get("wav_path")
    if isinstance(wp, str) and wp.strip():
        cands.append(Path(wp).name)

    spk = rec.get("speaker_id")
    uid = rec.get("utterance_id")
    if isinstance(spk, str) and spk.strip():
        cands.append(spk)
        spk_std = spk.replace("copy of ", "copy_of_").replace("Copy of ", "Copy_of_").replace(" ", "_")
        cands.append(spk_std)
    if isinstance(uid, str) and uid.strip():
        cands.append(ensure_wav_name(uid))
        uid_std = uid.replace("copy of ", "copy_of_").replace("Copy of ", "Copy_of_").replace(" ", "_")
        cands.append(ensure_wav_name(uid_std))


    uniq = []
    seen = set()
    for name in cands:
        if not isinstance(name, str):
            continue
        base = ensure_wav_name(Path(name).name)
        if base not in seen:
            uniq.append(base)
            seen.add(base)
    return uniq

def resolve_audio_path(rec: Dict, audio_root: Path,
                       exact_by_name: Dict[str, Path],
                       fuzzy_by_key: Dict[str, List[Path]]) -> Optional[Path]:


    wp = rec.get("wav_path")
    if isinstance(wp, str) and wp.strip():
        p = Path(wp)
        if p.is_file():
            return p.resolve()
        p2 = (audio_root / p.name)
        if p2.is_file():
            return p2.resolve()


    for cand_name in generate_name_candidates(rec):

        cand_path = (audio_root / cand_name)
        if cand_path.is_file():
            return cand_path.resolve()

        if cand_name in exact_by_name:
            return exact_by_name[cand_name].resolve()

        k = simplify_for_match(cand_name)
        hits = fuzzy_by_key.get(k, [])
        if hits:

            hits_sorted = sorted(hits, key=lambda x: len(str(x)))
            return hits_sorted[0].resolve()

    return None

def safe_copy(src: Path, dst_dir: Path, dst_name: Optional[str] = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_name is None:
        dst_path = dst_dir / src.name
    else:
        dst_path = dst_dir / dst_name

    if dst_path.exists():
        stem = dst_path.stem
        suf  = dst_path.suffix
        k = 1
        while True:
            alt = dst_dir / f"{stem}__{k}{suf}"
            if not alt.exists():
                dst_path = alt
                break
            k += 1
    shutil.copy2(src, dst_path)
    return dst_path

def main():
    parser = argparse.ArgumentParser(
        description="Thu thập audio dự đoán sai dựa trên checkpoints/check_results/misclassified.jsonl"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="checkpoints/check_results/misclassified.jsonl",
        help="Đường dẫn file JSONL chứa danh sách mẫu dự đoán sai."
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default="wavs16k",
        help="Thư mục gốc chứa file audio (*.wav). Mặc định: wavs16k"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="checkpoints/check_results/misclassified_wavs",
        help="Thư mục đích để copy các file audio dự đoán sai."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="err",
        help="Tiền tố file đích để dễ phân nhóm (vd: err_)."
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    audio_root = Path(args.audio_root).resolve()
    outdir     = Path(args.outdir).resolve()

    if not jsonl_path.exists():
        print(f"[ERR] Không tìm thấy JSONL: {jsonl_path}")
        sys.exit(1)
    if not audio_root.exists():
        print(f"[ERR] Không thấy audio_root: {audio_root}")
        sys.exit(1)


    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] JSONL:      {jsonl_path}")
    print(f"[INFO] AUDIO_ROOT: {audio_root}")
    print(f"[INFO] OUTDIR:     {outdir}")

    items = read_jsonl(jsonl_path)
    if not items:
        print("[WARN] JSONL không có dòng hợp lệ.")
        sys.exit(0)


    print("[INFO] Đang quét danh sách .wav trong audio_root (có thể mất vài giây)...")
    wav_files = list_all_wavs(audio_root)
    exact_by_name, fuzzy_by_key = build_wav_index(wav_files)
    print(f"[INFO] Tìm thấy {len(wav_files)} file .wav")

    copied = []
    missing = []

    for i, rec in enumerate(items, start=1):
        true_lb = rec.get("true_label", str(rec.get("true_label_id", "NA")))
        pred_lb = rec.get("pred_label", str(rec.get("pred_label_id", "NA")))
        uid     = rec.get("utterance_id", f"item{i:04d}")
        conf    = rec.get("confidence", 0.0)
        basename = Path(rec.get("speaker_id") or f"{uid}.wav").name

        dst_name = f"{args.prefix}_{i:04d}__{true_lb}-{pred_lb}__{basename}"

        src_path = resolve_audio_path(rec, audio_root, exact_by_name, fuzzy_by_key)
        if src_path is None or not src_path.exists():
            missing.append({
                "index": i,
                "utterance_id": uid,
                "speaker_id": rec.get("speaker_id"),
                "candidates_tried": generate_name_candidates(rec),
                "note": "Không tìm thấy file audio trong audio_root",
            })
            print(f"[MISS] {i:04d} | {uid} | {rec.get('speaker_id')}")
            continue

        dst_path = safe_copy(src_path, outdir, dst_name=dst_name)
        copied.append({
            "index": i,
            "utterance_id": uid,
            "speaker_id": rec.get("speaker_id"),
            "src": str(src_path),
            "dst": str(dst_path),
            "true_label": true_lb,
            "pred_label": pred_lb,
            "confidence": float(conf),
        })
        print(f"[COPY] {i:04d} | {true_lb}->{pred_lb} | {dst_path.name}")


    if copied:
        csv_path = outdir / "copied_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "index", "utterance_id", "speaker_id", "src", "dst",
                "true_label", "pred_label", "confidence"
            ])
            w.writeheader()
            for row in copied:
                w.writerow(row)
        print(f"[OK] Summary copied CSV: {csv_path}")

        json_path = outdir / "copied_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(copied, f, ensure_ascii=False, indent=2)
        print(f"[OK] Summary copied JSON: {json_path}")
    else:
        print("[NOTE] Không copy được file nào.")

    if missing:
        miss_path = outdir / "missing.json"
        with miss_path.open("w", encoding="utf-8") as f:
            json.dump(missing, f, ensure_ascii=False, indent=2)
        print(f"[NOTE] Missing list: {miss_path}")

    print(f"[DONE] Tổng số dòng: {len(items)} | Copied: {len(copied)} | Missing: {len(missing)}")

if __name__ == "__main__":
    main()
