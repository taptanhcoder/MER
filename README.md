# VietMER: Vietnamese Multimodal Emotion Recognition (Text + Audio)

**VietMER** is a research-oriented framework for **Multimodal Emotion Recognition (MER)** from **Vietnamese speech and transcripts**. It is built to support **academic experimentation**, **reproducible benchmarking**, and practical prototyping for **call center intelligence** and related affect-aware applications.

---

## Introduction

VietMER is a modular **encoder → fusion → classifier** framework for MER, designed to jointly model:
- **Lexical semantics** from Vietnamese text (transcripts), and  
- **Paralinguistic cues** from speech audio (prosody, rhythm, energy).

By preserving **sequence-level representations** (token/frame sequences) and providing mask-correct fusion/pooling, VietMER enables systematic study of pretrained text and speech encoders and their cross-modal interactions under realistic **variable-length** settings.

While the project targets academic research and reproducibility, it is also motivated by practical deployment in **call center intelligence**, where detecting customer emotion/attitude can support agent assistance, escalation prevention, QA automation, and conversation analytics.

---

## Research Goals

- **Multimodal Representation Learning for Vietnamese**  
  Study complementary contributions of Vietnamese text encoders (**PhoBERT / ViDeBERTa**) and speech encoders (**wav2vec2 / Whisper / Fourier2Vec**).

- **Cross-Modal Interaction Modeling**  
  Evaluate fusion strategies (bidirectional **cross-attention** and recurrent baselines) for alignment, robustness, and modality reliance.

- **Mask-Correct Learning under Variable-Length Signals**  
  Ensure correct masking across encoder time-scales (notably for downsampled audio encoders) to prevent padding leakage in attention and pooling.

- **Reproducible Benchmarking & Extensions**  
  Provide a consistent experimental platform for ablations (encoder/fusion/pooling/sampling), comparative evaluation, and future research modules.

---

## Applications

### Primary: Call Center Emotion & Attitude Analytics
- **Turn-level** emotion/attitude classification and **call-level** trend aggregation  
- **Real-time escalation risk** estimation (e.g., rising anger/frustration over time)  
- **Agent assistance** prompts and **auto-routing** decisions  
- **Post-call QA** tagging, analytics dashboards, and monitoring

### Additional Use-Cases
- Voice assistants and conversational agents (adaptive response strategies)  
- E-learning engagement monitoring (confusion/frustration detection)  
- Meeting analytics (tension/conflict trend estimation, with diarization)  
- In-car voice interaction (stress-aware UI adaptation)  
- Other affect-aware spoken-language applications

---

## Open-Source Impact

VietMER is intended as an open-source library enabling both researchers and developers to integrate multimodal emotion understanding into downstream systems. The repository emphasizes:
- **Modularity** (encoders/fusion strategies are swappable),
- **Extensibility** (research prototypes can be integrated cleanly),
- **Reproducibility** (config-driven experiments and consistent training/evaluation utilities).

---

## Method Summary (Architecture)

### Encoder → Fusion → Classifier
- **Text Encoder**: PhoBERT / ViDeBERTa (Transformer encoder-only; token-level sequence)
- **Audio Encoder** (selectable):
  - wav2vec2 XLSR-53
  - Whisper encoder wrapper
  - Fourier2Vec (STFT → log-mel → Transformer)
  - Dual wav2vec2 + VGGish
- **Fusion** (selectable):
  - Bidirectional Cross-Attention
  - BiLSTM + Cross-Attention Blocks
  - Temporal CNN + BiLSTM + Cross-Attention Blocks
- **Pooling**: `cls`, masked `mean/max/min`, or learnable attention pooling
- **Classifier**: configurable MLP head → emotion logits

---

## Data Interface

The code expects JSONL splits (e.g., `output/train.jsonl`, `output/valid.jsonl`, `output/test.jsonl`).  
Each sample includes waveform path, segment boundaries, transcript, and emotion label:

```json
{
  "utterance_id": "utt_001",
  "speaker_id": "spk_01",
  "wav_path": "wavs16k/example.wav",
  "start": 0.0,
  "end": 10.40,
  "transcript": "...",
  "emotion": "angry"
}
```


## Training & Evaluation

- Mixed precision training (AMP) + gradient clipping

- Checkpointing: periodic + best-on-validation

- Efficient batching: length-bucket batching (audio+text mixed length) and/or weighted sampling

- Loss functions: Cross-Entropy / Focal Loss / Label Smoothing

- Metrics: Accuracy + Macro-F1 (recommended under class imbalance)

## Reproducibility & Ablation Suggestions

- VietMER is designed for controlled ablations:

- Encoders: PhoBERT vs ViDeBERTa; wav2vec2 vs Whisper vs Fourier2Vec vs Dual

- Fusion: xattn vs bilstm_attn vs cnn_bilstm_attn

- Pooling: cls vs mean/max/min vs attn pooling

- Sampling: weighted sampler vs length bucket vs both