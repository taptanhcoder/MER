
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="Fsoft-AIC/videberta-base",
                        help="Tên repo trên Hugging Face (org/name).")
    parser.add_argument("--dst", type=str, default="MER/videberta-base",
                        help="Thư mục đích lưu checkpoint (local mirror).")
    parser.add_argument("--use-auth", action="store_true",
                        help="Dùng auth token (cho repo private). Yêu cầu `huggingface-cli login` trước.")
    parser.add_argument("--force-download", action="store_true",
                        help="Bỏ qua cache, tải lại (nếu cần).")
    args = parser.parse_args()

    dst_dir = Path(args.dst).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Mirror repo: {args.repo}")
    print(f"[+] Local dir  : {dst_dir}")

    local_path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(dst_dir),
        local_dir_use_symlinks=False,
        use_auth_token=True if args.use_auth else None,
        revision=None,  
        ignore_patterns=["*.msgpack", "*.h5", "*.safetensors_tmp*"],  
        force_download=args.force_download,
    )

    print(f"[✓] Downloaded to: {local_path}")


    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        cfg = AutoConfig.from_pretrained(local_path, output_hidden_states=True, output_attentions=False)
        tok = AutoTokenizer.from_pretrained(local_path, use_fast=True)
        mdl = AutoModel.from_pretrained(local_path, config=cfg)
        hs = getattr(mdl.config, "hidden_size", None)
        print(f"[✓] Load OK from local. hidden_size={hs}")
    except Exception as e:
        print("[!] WARNING: Cannot load model/tokenizer from local path.")
        print("    → Kiểm tra lại files trong thư mục mirror hoặc dependency transformers.")
        print("    → Error:", repr(e))

    print("\nGợi ý cấu hình training MER:")
    print(f"  cfg.text_encoder_type = 'videberta'")
    print(f"  cfg.text_encoder_ckpt = r'{str(dst_dir)}'")
    print("\nDone.")

if __name__ == "__main__":
    main()
