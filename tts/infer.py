import subprocess
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import tensorflow as tf
import pickle

import time

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).parent.parent

ASSETS = ROOT / "assets"
MODELS = ASSETS / "models"

TMP = Path(__file__).parent / "tmp"
TMP.mkdir(exist_ok=True)

SR = 16000
FRAME_PERIOD = 0.005

# =========================
# LOAD MODELS
# =========================

duration_model = tf.keras.models.load_model(
    MODELS / "duration.keras"
)

acoustic_model = tf.keras.models.load_model(
    MODELS / "acoustic.keras"
)

with open(MODELS / "encoder.pkl", "rb") as f:
    phone_encoder = pickle.load(f)

# =========================
# G2P (MFA)
# =========================

def text_to_phonemes(text):

    input_txt = TMP / "input.txt"
    output_txt = TMP / "output.txt"

    input_txt.write_text(text, encoding="utf-8")

    cmd = f"""
    source ~/miniforge3/etc/profile.d/conda.sh && \
    conda activate mfa && \
    mfa g2p \
    "{input_txt}" \
    ukrainian_mfa \
    "{output_txt}"
    """

    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)

    phones_set = set()
    phones = ["pau"]

    with open(output_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            tuple_word = tuple(parts[0])
            if tuple_word not in phones_set:
                phones_set.add(tuple_word)
                phones.extend(parts[1:])
                phones.append("pau")

    # phones.pop()

    return phones

# =========================
# CONTEXT BUILD
# =========================

def build_context(phones):

    contexts = []

    for i in range(len(phones)):

        prev_phone = phones[i - 1] if i > 0 else "<bos>"
        cur_phone = phones[i]
        next_phone = phones[i + 1] if i < len(phones) - 1 else "<eos>"

        contexts.append([prev_phone, cur_phone, next_phone])

    return contexts

# =========================
# ENCODE CONTEXT
# =========================

def encode_context(contexts):

    X = []

    for ctx in contexts:
        try:
            X.append(phone_encoder.transform(ctx))
        except ValueError:
            # fallback for unknown phones
            X.append(phone_encoder.transform([
                "<bos>" if x not in phone_encoder.classes_ else x
                for x in ctx
            ]))

    return np.array(X, dtype=np.int32)

# =========================
# DURATION PREDICTION
# =========================

def predict_durations(X):

    preds = duration_model.predict(X, verbose=0).reshape(-1)

    preds = np.maximum(preds, 0.03)  # clamp

    return preds

# =========================
# MAKE LABELS
# =========================

def make_labels(phones, durations):

    labels = []

    current = 0

    for phone, dur in zip(phones, durations):

        start = current
        end = current + int(dur * 1e7)

        labels.append((start, end, phone))

        current = end

    return labels

# =========================
# ACOUSTIC PREDICTION
# =========================

def predict_acoustic(X, durations):

    phones = []
    positions = []

    f0_frames = []
    spec_frames = []

    for i in range(len(X)):

        frame_count = max(
            1,
            int(round(durations[i] / FRAME_PERIOD))
        )

        for frame_idx in range(frame_count):
            if frame_count <= 1:
                rel_pos = 0.0
            else:
                rel_pos = frame_idx / (frame_count - 1)

            phones.append(X[i])
            positions.append([rel_pos])

    phones = np.asarray(phones, dtype=np.int32)
    positions = np.asarray(positions, dtype=np.float32)

    print("phones:", phones.shape)
    print("positions:", positions.shape)

    preds = acoustic_model.predict(
        {
            "phones": phones,
            "position": positions
        },
        verbose=0
    )

    pau_id = phone_encoder.transform(["pau"])[0]

    for i, pred in enumerate(preds):

        pred = np.asarray(pred, dtype=np.float64)

        pred_f0 = float(pred[0])

        if pred_f0 < 30.0:
        # if pred_f0 < 0.0:
            f0_value = 0.0
        else:
            f0_value = pred_f0

        spec_value = pred[1:]

        if len(spec_value) != 513:
            raise RuntimeError(
                f"Expected 513 bins, got {len(spec_value)}"
            )
        
        if phones[i][1] == pau_id:
            f0_value = 0.0
            spec_value.fill(1e-6)

        f0_frames.append(f0_value)
        spec_frames.append(spec_value)

    return (
        np.asarray(f0_frames, dtype=np.float64),
        np.asarray(spec_frames, dtype=np.float64)
    )

# =========================
# SYNTHESIS (WORLD)
# =========================

def synthesize(f0, spectrogram):

    if len(f0) == 0:
        raise RuntimeError("Empty f0")

    spectrogram = np.nan_to_num(spectrogram)
    f0 = np.nan_to_num(f0)

    spectrogram = np.maximum(spectrogram, 1e-6)

    ap = np.ones_like(spectrogram) * 0.001

    wav = pw.synthesize(
        f0,
        spectrogram,
        ap,
        SR,
        FRAME_PERIOD * 1000.0
    )

    return wav

# =========================
# FILTER
# =========================

from scipy.signal import butter, filtfilt

def lowpass_filter(wav, cutoff=6000):

    nyquist = SR / 2

    b, a = butter(
        4,
        cutoff / nyquist,
        btype="low"
    )

    return filtfilt(b, a, wav)

# =========================
# MAIN
# =========================

def main():

    text = input("TEXT > ").strip()

    if not text:
        return
    
    text = text.lower()

    print("\nG2P...")
    phones = text_to_phonemes(text)

    print("\nPHONES:", phones)

    print("\nBuilding context...")
    contexts = build_context(phones)

    print("\nEncoding...")
    X = encode_context(contexts)

    print("\nPredicting durations...")
    durations = predict_durations(X)

    labels = make_labels(phones, durations)

    print("\nLABELS:")
    for l in labels:
        print(l)

    start = time.time()

    print("\nPredicting acoustic...")
    f0, spec = predict_acoustic(X, durations)

    # from scipy.ndimage import gaussian_filter1d

    # f0 = gaussian_filter1d(f0, sigma=3)

    # spec = gaussian_filter1d(
    #     spec,
    #     sigma=1.5,
    #     axis=0
    # )

    print("\nSynthesizing...")
    wav = synthesize(f0, spec)

    end = time.time()

    print(f"Execution take {end-start} seconds")

    # print("\nFiltering...")
    # wav = lowpass_filter(wav)

    out_path = Path(__file__).parent / "audio" / f"{text}.wav"

    sf.write(out_path, wav, SR)

    print("\nDONE:", out_path)


if __name__ == "__main__":
    main()

# import soundfile as sf

# def synthesize_text(text, output_path):

#     text = text.lower().strip()

#     phones = text_to_phonemes(text)
#     contexts = build_context(phones)
#     X = encode_context(contexts)

#     durations = predict_durations(X)
#     f0, spec = predict_acoustic(X, durations)

#     wav = synthesize(f0, spec)
#     wav = lowpass_filter(wav)

#     sf.write(output_path, wav, SR)

#     return output_path