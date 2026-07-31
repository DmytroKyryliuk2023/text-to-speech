# Ukrainian Text-to-Speech (TTS)

A lightweight research project for building a Ukrainian Text-to-Speech (TTS) system using TensorFlow, WORLD vocoder, and Montreal Forced Aligner (MFA).

The project provides a complete pipeline for:

- dataset preprocessing;
- forced phoneme alignment;
- HTS label generation;
- duration model training;
- acoustic model training;
- speech synthesis from text.

This project is intended for research and educational purposes rather than production deployment.

---

# Repository Structure

```
.
├── assets/
│   ├── corpus/          # Preprocessed dataset for MFA
│   ├── alignments/      # MFA TextGrid alignments
│   ├── labels/          # HTS label files
│   └── models/          # Trained models
│
├── dataset/
│   ├── metadata.csv
│   └── wavs/
│
├── tts/
│   ├── prepare.py
│   ├── train.py
│   ├── infer.py
│   ├── audio/
│   └── tmp/
│
└── README.md
```

---

# Requirements

- Python 3.10+
- TensorFlow
- Montreal Forced Aligner (MFA)
- Ukrainian MFA acoustic model
- Ukrainian MFA pronunciation dictionary

Python dependencies:

```bash
pip install -r requirements.txt
```

---

# Installing Montreal Forced Aligner

Install MFA according to the official documentation.

After installation, download the Ukrainian acoustic model and pronunciation dictionary.

The project expects the following MFA resources:

- `ukrainian_mfa` acoustic model
- `ukrainian_mfa` pronunciation dictionary

The scripts automatically call MFA through:

```bash
mfa align
```

and

```bash
mfa g2p
```

---

# Dataset Format

```
dataset/
├── metadata.csv
└── wavs/
    ├── 000001.wav
    ├── 000002.wav
    └── ...
```

Example `metadata.csv`:

```
000001|Привіт світе
000002|Це приклад речення
```

---

# Workflow

```
Dataset
   │
   ▼
prepare.py
   │
   ▼
Forced alignment (MFA)
   │
   ▼
HTS labels
   │
   ▼
train.py
   │
   ▼
TensorFlow models
   │
   ▼
infer.py
   │
   ▼
Generated speech
```

---

# Usage

## 1. Prepare dataset

```bash
python tts/prepare.py
```

This script:

- normalizes audio to 16 kHz;
- cleans transcription text;
- runs Montreal Forced Aligner;
- converts TextGrid files into HTS label files.

Generated files:

```
assets/corpus/
assets/alignments/
assets/labels/
```

---

## 2. Train models

```bash
python tts/train.py
```

The training script:

- extracts WORLD features;
- trains a duration prediction model;
- trains an acoustic model;
- saves TensorFlow models and the phone encoder.

Models are stored in:

```
assets/models/
```

Generated files:

```
duration.keras
acoustic.keras
encoder.pkl
```

---

## 3. Run inference

```bash
python tts/infer.py
```

The inference pipeline performs:

1. Grapheme-to-phoneme conversion (MFA)
2. Context generation
3. Duration prediction
4. Acoustic feature prediction
5. WORLD speech synthesis

Generated audio is saved into:

```
tts/audio/
```

---

# Models

## Duration Model

Input:

- previous phoneme
- current phoneme
- next phoneme

Output:

- phoneme duration

Architecture:

- Embedding
- Bidirectional LSTM
- Dense layers

---

## Acoustic Model

Input:

- phoneme context
- relative frame position

Output:

- F0
- WORLD spectral envelope

Architecture:

- Embedding
- Bidirectional LSTM
- Fully-connected layers

---

# Output Directories

```
assets/
├── corpus/
├── alignments/
├── labels/
└── models/

tts/
├── audio/
└── tmp/
```

---

# Notes

- Audio is resampled to **16 kHz**.
- WORLD is used as the vocoder.
- TensorFlow is used for both neural networks.
- MFA performs forced alignment and grapheme-to-phoneme conversion.
- Unknown phonemes are replaced during encoding.

---

# License

This project is intended for research and educational use.