# src/train_mer_videberta_w2v2vgg_cnn_bilstm.py
import os
from pathlib import Path
import json
import torch

from train import Trainer, compute_class_weights_lenfreq   # tái dùng logic AMP/clip-grad/callback
from configs.base import Config
from loading.dataloader import build_train_test_dataset, VNEMOSDataset
from model.networks import MER
from model.losses import CrossEntropyLoss
from training.callbacks import CheckpointsCallback
from training.optimizers import split_param_groups, build_optimizer
from transformers import get_cosine_schedule_with_warmup, BatchEncoding


def main():
    # === Run config ===
    ckpt_dir = Path("checkpoints/videberta_w2v2vgg_cnn_bilstm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        name="MER_ViDeBERTa_W2V2VGG_CNNBiLSTM",
        checkpoint_dir=str(ckpt_dir),
        num_epochs=50,
        batch_size=8,
        num_workers=2,

        # Optim
        learning_rate=2e-4,             # LR cho head; encoder dùng LR nhỏ hơn qua split_param_groups()
        optimizer_type="AdamW",
        scheduler_type="cosine_warmup",
        warmup_ratio=0.05,
        scheduler_step_unit="step",

        # Loss
        loss_type="CrossEntropyLoss",
        label_smoothing=0.0,

        # Data
        data_root="output",
        jsonl_dir="",
        sample_rate=16000,
        max_audio_sec=None,
        text_max_length=96,

        # Samplers
        use_length_bucket=True,
        length_bucket_size=64,
        bucketing_text_alpha=0.03,
        use_weighted_sampler=True,
        lenfreq_alpha=0.5,

        # ===== MODEL =====
        model_type="MemoCMT",

        # Text encoder → ViDeBERTa
        text_encoder_type="videberta",           # {"phobert","videberta"}
        text_encoder_ckpt="../videberta-base",   # hoặc "Fsoft-AIC/videberta-base" nếu không có local
        text_encoder_dim=768,
        text_unfreeze=False,

        # Audio encoder → Dual W2V2 + VGGish
        # CHÍNH XÁC khóa: "w2v2_vggish" (user hay gõ nhầm "Wac2vec_VGG")
        audio_encoder_type="w2v2_vggish",
        audio_encoder_ckpt="facebook/wav2vec2-large-xlsr-53",   # VGGish dùng weights nội bộ adapter
        audio_encoder_dim=1024,                                  # sẽ overwrite runtime
        audio_unfreeze=False,
        w2v2_vggish_merge="concat",                              # {"concat","sum"}

        # Fusion = CNN-Bi-LSTM Attention
        fusion_type="cnn_bilstm_attn",           # cần case này đã được hỗ trợ trong networks.py
        fusion_dim=768,
        fusion_head_output_type="max",           # {'cls','mean','max','min','attn'}
        linear_layer_output=[256, 128],
        dropout=0.10,

        # Hyper cho CNN trước BiLSTM (có thể tinh chỉnh)
        fusion_cnn_channels=768,                 # out channels của Conv1d (thường = fusion_dim)
        fusion_cnn_kernel=3,
        fusion_cnn_dilations=[1, 2, 4],          # các block giãn nở (dilated) để gom ngữ cảnh

        # Hyper BiLSTM sau CNN
        fusion_bilstm_hidden_text=384,
        fusion_bilstm_hidden_audio=384,
        fusion_bilstm_layers=1,
        fusion_bilstm_dropout=0.1,
        fusion_bilstm_bidirectional=True,
        fusion_blocks=1,                          # số block cross-attn 2 chiều sau BiLSTM
        fusion_merge="concat",                    # {"concat","gate"}  (gating optional)
        fusion_pool_heads=1,                      # nếu dùng pooling 'attn'

        # Tricks
        use_amp=True,
        max_grad_norm=1.0,

        save_best_val=True,
        max_to_keep=2,
    )

    # === Data ===
    train_loader, eval_loader = build_train_test_dataset(cfg)
    _train_ds = VNEMOSDataset(cfg, "train")
    label2id = _train_ds.label2id
    id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
    cfg.num_classes = len(id2label)

    # === Model ===
    device = "cuda" if (torch.cuda.is_available() and os.getenv("USE_CUDA", "1") != "0") else "cpu"
    net = MER(cfg, device=device).to(device)

    # === Class weights (len-freq) ===
    class_weights = compute_class_weights_lenfreq(_train_ds, alpha=0.5).to(device)
    print("Class weights (len-freq):", class_weights.tolist())
    criterion = CrossEntropyLoss(cfg, weight=class_weights)

    trainer = Trainer(cfg, net, criterion, log_dir="logs")

    # === Optimizer & Scheduler ===
    enc_lr, head_lr = 5e-6, float(cfg.learning_rate)        # encoder nhỏ; head theo cfg
    param_groups = split_param_groups(trainer, lr_enc=enc_lr, lr_head=head_lr, weight_decay=0.05)
    optimizer = build_optimizer("adamw", param_groups, lr=head_lr, weight_decay=0.05)

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    trainer.compile(
        optimizer=optimizer,
        scheduler=scheduler,
        lr=head_lr,
        param_groups=None,
        scheduler_step_unit=cfg.scheduler_step_unit
    )

    # === Checkpoint callback (best theo val_macro_f1) ===
    ckpt_cb = CheckpointsCallback(
        cfg.checkpoint_dir,
        save_freq=200,
        max_to_keep=cfg.max_to_keep,
        save_best_val=True,
        monitor="val_macro_f1",
        mode="max",
    )
    callbacks = [ckpt_cb]

    # === Train ===
    trainer.fit(train_loader, epochs=cfg.num_epochs, eval_data=eval_loader, callbacks=callbacks)
    best_path = getattr(ckpt_cb, "best_path", "")
    print("Best checkpoint:", best_path)

    # (optional) load best weights
    if isinstance(best_path, str) and best_path.endswith(".pth") and Path(best_path).exists():
        print("Loading best weights:", best_path)
        state = torch.load(best_path, map_location=trainer.device)
        trainer.network.load_state_dict(state)

    # === Evaluate & Report ===
    def collect_preds(_trainer, loader):
        all_preds, all_labels = [], []
        _trainer.network.eval()
        with torch.no_grad():
            for batch in loader:
                tok, audio, labels, meta = _trainer._unpack_batch(batch)
                audio  = audio.to(_trainer.device)
                labels = labels.to(_trainer.device)
                # move text to device
                if isinstance(tok, BatchEncoding):
                    tok = tok.to(_trainer.device); tok = {k: v for k, v in tok.items()}
                elif isinstance(tok, dict):
                    tok = {k: v.to(_trainer.device) for k, v in tok.items()}
                else:
                    tok = tok.to(_trainer.device)

                out = _trainer.network(tok, audio, meta=meta)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_preds), torch.cat(all_labels)

    if eval_loader is not None:
        preds, labels = collect_preds(trainer, eval_loader)
        num_classes = cfg.num_classes

        cm = torch.zeros((num_classes, num_classes), dtype=torch.int32)
        for t, p in zip(labels, preds):
            cm[t.long(), p.long()] += 1

        per_class_acc = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()).numpy()
        overall_acc = (preds == labels).float().mean().item()

        print("=== Evaluation ===")
        print("Overall Acc: %.4f" % overall_acc)
        print("Per-class Acc:")
        id2label = [k for k, v in sorted(label2id.items(), key=lambda x: x[1])]
        for i, acc in enumerate(per_class_acc):
            name = id2label[i] if i < len(id2label) else str(i)
            print(f"  {i} ({name}): {acc:.4f}")
        print("Confusion Matrix:\n", cm.numpy())

        try:
            from sklearn.metrics import classification_report
            target_names = [id2label[i] for i in range(num_classes)]
            print("\nClassification Report:")
            print(classification_report(labels.numpy(), preds.numpy(), target_names=target_names, digits=4))
        except Exception as e:
            print("sklearn not available:", e)
    else:
        print("No eval_loader available.")

    # === Export HDF5 (.h5) ===
    import h5py
    h5_name = "model_weights_videberta_w2v2vgg_cnn_bilstm.h5"
    h5_path = ckpt_dir / h5_name

    state = trainer.network.state_dict()
    torch_ver = str(getattr(torch, "__version__", "unknown"))
    id2label_json = json.dumps([k for k, v in sorted(label2id.items(), key=lambda x: x[1])], ensure_ascii=False)

    with h5py.File(h5_path, "w") as f:
        f.attrs["model_name"] = "MER"
        f.attrs["num_classes"] = int(cfg.num_classes)
        f.attrs["id2label"] = id2label_json
        f.attrs["torch_version"] = torch_ver

        cfg_group = f.create_group("config")
        for k, v in vars(cfg).items():
            try:
                cfg_group.attrs[k] = json.dumps(v, ensure_ascii=False)
            except TypeError:
                cfg_group.attrs[k] = str(v)

        weights_group = f.create_group("state_dict")
        for name, tensor in state.items():
            arr = tensor.detach().cpu().numpy()
            dset = weights_group.create_dataset(name, data=arr)
            dset.attrs["shape"] = arr.shape
            dset.attrs["dtype"] = str(arr.dtype)

    print(f" Saved HDF5 weights to: {h5_path.resolve()}")


if __name__ == "__main__":
    main()
