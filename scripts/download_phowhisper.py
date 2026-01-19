import argparse
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[!] Missing dependency: huggingface_hub\n    pip install -U huggingface_hub", file=sys.stderr)
        raise

    parser = argparse.ArgumentParser("Mirror a Whisper / Pho-Whisper repo to local folder")
    parser.add_argument("--repo", required=True,
                        help="Hugging Face repo id, e.g. openai/whisper-small or <org>/Pho-Whisper-xx")
    parser.add_argument("--dst", required=True,
                        help="Local folder to store the checkpoint, e.g. MER/phowhisper-small")
    parser.add_argument("--revision", default=None, help="Optional commit/tag/branch")
    parser.add_argument("--force", action="store_true", help="Delete dst first if exists")
    args = parser.parse_args()

    dst = Path(args.dst).expanduser().resolve()
    if args.force and dst.exists():
        import shutil
        print(f"[i] --force: removing existing folder: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[+] Mirror repo: {args.repo}")
    print(f"[+] Local dir  : {dst}")

    # Download all files needed for model+processor
    try:
        local_path = snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=str(dst),
            local_dir_use_symlinks=False,  
            allow_patterns=None,          
            tqdm_class=None,
        )
        print(f"[✓] Downloaded to: {local_path}")
    except Exception as e:
        print(f"[x] Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    ok_model = False
    try:
        from transformers import WhisperModel, WhisperFeatureExtractor
        mdl = WhisperModel.from_pretrained(str(dst))
        fe  = WhisperFeatureExtractor.from_pretrained(str(dst))
        print("[✓] Load OK:")
        print(f"    model_type: {getattr(mdl.config, 'model_type', 'N/A')}")
        print(f"    d_model   : {getattr(mdl.config, 'd_model', 'N/A')}")
        print(f"    num_mels  : {getattr(fe, 'n_mels', 'N/A')}, sr: {getattr(fe, 'sampling_rate', 'N/A')}")
        ok_model = True
    except Exception as e:

        print(
            "[!] WARNING: Cannot load model/feature-extractor from local path.\n"
            "    → Kiểm tra transformers/torch version hoặc định dạng weights (safetensors/bin).\n"
            f"    → Error: {repr(e)}"
        )


    print("\nGợi ý cấu hình training MER:")
    print("  cfg.audio_encoder_type = 'phowhisper'")
    print(f"  cfg.audio_encoder_ckpt = r'{dst}'")
    if ok_model:
        print("  # PhoWhisperEncoder sẽ đọc hidden_size từ model.config.d_model")

if __name__ == "__main__":
    main()
