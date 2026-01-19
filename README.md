# MER-VN: A Research-Oriented Framework for Vietnamese Multimodal Emotion Recognition (Text + Audio)

A modular, academic-first codebase for **Multimodal Emotion Recognition (MER)** from **Vietnamese transcripts and speech**.  
MER-VN is designed for **systematic research**, enabling controlled ablations over **encoders**, **fusion mechanisms**, **mask-aware pooling**, and **training/sampling protocols**.

---

## Research Goals

MER-VN aims to study and improve MER performance by:
- Leveraging **pretrained self-supervised encoders** for both text and audio
- Modeling **cross-modal interactions** via learnable fusion layers
- Ensuring **sequence-level** representations (not only `[CLS]`) to preserve temporal alignment
- Supporting **reproducible experimentation** with configurable training and evaluation

---

## Core Contributions (Project-Level)

1. **Modular Encoder–Fusion–Classifier Design**  
   A plug-and-play architecture that allows easy replacement of text/audio encoders and fusion modules for research comparisons.

2. **Sequence-Preserving Multimodal Modeling**  
   The system retains full token/frame sequences:
   - Text: \( \mathbf{T}\in\mathbb{R}^{B\times T_t\times d_t} \)
   - Audio: \( \mathbf{A}\in\mathbb{R}^{B\times T_a\times d_a} \)  
   enabling fine-grained fusion beyond pooled embeddings.

3. **Mask-Correct Fusion for Downsampled Audio Encoders**  
   Audio padding masks are **reconstructed at the encoder output time-scale** using `get_feat_lengths()` (for wav2vec2/Fourier2Vec/Whisper wrappers when available), preventing padding leakage in attention and pooling.

4. **Research-Friendly Training Efficiency**  
   - **Length-bucket batching** using a mixed length proxy:
     \[
       \ell_{mix}=\ell_{audio(sec)}+\alpha\cdot \ell_{text(words)}
     \]
   - **Weighted sampling** that adjusts for both **class imbalance** and **length bias**

5. **Multi-Strategy Fusion Suite**  
   Provides multiple fusion families for ablation:
   - Bidirectional **Cross-Attention** (text↔audio)
   - **BiLSTM + Cross-Attention Blocks**
   - **Temporal CNN + BiLSTM + Cross-Attention Blocks**

---

## Method Overview

### Problem Definition
Given an utterance with transcript \(x^{(t)}\), waveform \(x^{(a)}\), and label \(y\in\{1,\dots,C\}\), learn:
\[
f_\theta(x^{(t)},x^{(a)}) \rightarrow \hat{y}
\]
under supervised objectives (Cross-Entropy / Focal / Label Smoothing).

### Architecture: Encoder → Fusion → Classifier

**1) Encoders**
- **Text**: PhoBERT / ViDeBERTa (Transformer encoder-only; full token sequence)
- **Audio** (selectable):
  - wav2vec2 XLSR-53 (self-supervised speech representations)
  - Whisper encoder wrapper (feature extractor → encoder)
  - Fourier2Vec (STFT → log-mel → Transformer encoder; lightweight)
  - Dual encoder (wav2vec2 + VGGish, merge by concat/sum)

**2) Projection to a Shared Fusion Space**
\[
\tilde{\mathbf{T}}=\mathrm{LN}(\mathbf{T}\mathbf{W}_t),\quad
\tilde{\mathbf{A}}=\mathrm{LN}(\mathbf{A}\mathbf{W}_a),\quad
\tilde{\mathbf{T}},\tilde{\mathbf{A}}\in\mathbb{R}^{B\times \cdot \times d}
\]

**3) Fusion (examples)**
- **Bidirectional Cross-Attention**:
\[
\mathbf{T}'=\mathrm{MHA}(\tilde{\mathbf{T}},\tilde{\mathbf{A}},\tilde{\mathbf{A}}),\quad
\mathbf{A}'=\mathrm{MHA}(\tilde{\mathbf{A}},\tilde{\mathbf{T}},\tilde{\mathbf{T}})
\]
\[
\mathbf{F}=[\mathbf{T}';\mathbf{A}']\in\mathbb{R}^{B\times(T_t+T_a)\times d}
\]

- **Recurrent baselines**: BiLSTM per modality → cross-attention blocks → optional gating → concat sequence

**4) Mask-Aware Pooling**
Pooling options for utterance representation \( \mathbf{z} \):
- `cls`, `mean`, `max`, `min` (masked reduce)
- `attn` (learnable attention pooling)

**5) Classifier**
Configurable MLP head \( \rightarrow \) logits \( \in \mathbb{R}^{B\times C} \).

---

## Data Interface

The pipeline expects JSONL splits (`train.jsonl`, `valid.jsonl`, `test.jsonl`). Each sample:
```json
{
  "utterance_id": "...",
  "speaker_id": "...",
  "wav_path": "wavs16k/....wav",
  "start": 0.0,
  "end": 10.4,
  "transcript": "...",
  "emotion": "angry"
}
Audio is cropped by [start, end], resampled to cfg.sample_rate.
Text is normalized (Unicode NFC + whitespace normalization).

Training & Evaluation (Research Protocol)
Trainer: AMP (mixed precision), gradient clipping, checkpointing (freq & best-on-val)

Sampling:

length-bucket batching (reduces padding across modalities)

weighted sampling (class + length bias)

Loss: Cross-Entropy / FocalLoss (default) / LabelSmoothingCE

Metrics: Accuracy + Macro-F1 (recommended for imbalance)

Reproducible Ablations (Suggested)
MER-VN is structured for clean ablation studies:

Encoders: PhoBERT vs ViDeBERTa; wav2vec2 vs Whisper vs Fourier2Vec vs Dual

Fusion: xattn vs bilstm_attn vs cnn_bilstm_attn

Pooling: cls vs mean/max/min vs attn pooling

Sampling: weighted vs length-bucket vs both

