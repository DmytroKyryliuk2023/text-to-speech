import pickle
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

# =========================
# PATHS
# =========================

ROOT = Path(__file__).parent.parent

WAV_DIR = ROOT / "assets" / "corpus"
LAB_DIR = ROOT / "assets" / "labels"

MODEL_DIR = ROOT / "assets" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

# =========================
# LOAD LABELS
# =========================

def load_labels(path):
    phones = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 3:
                continue

            start = int(parts[0]) / 10000000.0
            end = int(parts[1]) / 10000000.0
            phone = parts[2]

            duration = end - start

            phones.append((phone, start, end, duration))

    return phones


# =========================
# TRAIN DATA
# =========================

X_duration = []
y_duration = []

X_acoustic = []
y_acoustic = []

all_phones = []

wav_files = sorted(WAV_DIR.glob("*.wav"))
wav_files = wav_files[:1000]

print("FILES:", len(wav_files))

for wav_path in wav_files:

    stem = wav_path.stem
    lab_path = LAB_DIR / f"{stem}.lab"

    if not lab_path.exists():
        continue

    try:
        phones = load_labels(lab_path)

        audio, sr = librosa.load(wav_path, sr=SR)
        audio64 = audio.astype(np.float64)

        # WORLD
        f0, t = pw.dio(audio64, sr)
        f0 = pw.stonemask(audio64, f0, t, sr)
        sp = pw.cheaptrick(audio64, f0, t, sr)

        frame_period = 0.005

        # =========================
        # PHONE CONTEXT BUILD
        # =========================

        for i in range(len(phones)):

            phone, start, end, duration = phones[i]

            prev_phone = phones[i - 1][0] if i > 0 else "<bos>"
            next_phone = phones[i + 1][0] if i < len(phones) - 1 else "<eos>"

            context = [prev_phone, phone, next_phone]

            all_phones.extend(["<bos>", "<eos>", phone])

            # -----------------
            # duration model
            # -----------------
            X_duration.append(context)
            y_duration.append(duration)

            # -----------------
            # acoustic model
            # -----------------

            frame_start = int(start / frame_period)
            frame_end = int(end / frame_period)

            frame_start = max(0, frame_start)
            frame_end = min(len(sp), frame_end)

            if frame_end <= frame_start:
                continue

            frame_count = frame_end - frame_start

            for j in range(frame_start, frame_end):

                if frame_count <= 1:
                    rel_pos = 0.0
                else:
                    rel_pos = (
                        (j - frame_start)
                        / (frame_count - 1)
                    )

                cur_f0 = float(f0[j])

                target = np.concatenate(
                    (
                        [cur_f0],
                        sp[j]
                    )
                )

                X_acoustic.append(
                    (
                        context,
                        rel_pos
                    )
                )

                y_acoustic.append(target)

    except Exception as e:
        print("ERROR:", stem, e)


# =========================
# ENCODER
# =========================

print("Encoding phones...")

all_phones.extend(["<bos>", "<eos>"])

encoder = LabelEncoder()
encoder.fit(all_phones)

vocab_size = len(encoder.classes_)

X_duration_enc = np.array(
    [encoder.transform(x) for x in X_duration],
    dtype=np.int32
)

# new acoustic X
phone_features = []
position_features = []

for context, rel_pos in X_acoustic:

    phone_features.append(
        encoder.transform(context)
    )

    position_features.append(
        [rel_pos]
    )

phone_features = np.array(
    phone_features,
    dtype=np.int32
)

position_features = np.array(
    position_features,
    dtype=np.float32
)

y_duration = np.array(y_duration, dtype=np.float32)
y_acoustic = np.array(y_acoustic, dtype=np.float32)

print("Acoustic shape:", y_acoustic.shape)


# =========================
# DURATION MODEL (TF)
# =========================

print("\nTraining duration model...")

inputs = tf.keras.Input(shape=(3,))

x = tf.keras.layers.Embedding(vocab_size, 64)(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)

duration_model = tf.keras.Model(inputs, outputs)

duration_model.compile(
    optimizer="adam",
    loss="mse"
)

duration_model.fit(
    X_duration_enc,
    y_duration,
    epochs=50,
    batch_size=128,
    validation_split=0.1
)


# =========================
# ACOUSTIC MODEL (TF)
# =========================

print("\nTraining acoustic model...")

acoustic_dim = y_acoustic.shape[1]

phone_input = tf.keras.Input(
    shape=(3,),
    name="phones"
)

pos_input = tf.keras.Input(
    shape=(1,),
    name="position"
)

x = tf.keras.layers.Embedding(
    vocab_size,
    64
)(phone_input)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128)
)(x)

x = tf.keras.layers.Concatenate()(
    [x, pos_input]
)

x = tf.keras.layers.Dense(
    512,
    activation="relu"
)(x)

x = tf.keras.layers.Dense(
    512,
    activation="relu"
)(x)

outputs = tf.keras.layers.Dense(
    acoustic_dim
)(x)

acoustic_model = tf.keras.Model(
    [phone_input, pos_input],
    outputs
)

acoustic_model.compile(
    optimizer="adam",
    loss="mse"
)

acoustic_model.fit(
    {
        "phones": phone_features,
        "position": position_features,
    },
    y_acoustic,
    epochs=50,
    batch_size=1024,
    validation_split=0.1
)


# =========================
# SAVE
# =========================

print("\nSaving models...")

duration_model.save(MODEL_DIR / "duration.keras")
acoustic_model.save(MODEL_DIR / "acoustic.keras")

with open(MODEL_DIR / "encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


print("\nDONE")

print("phones:", len(all_phones))
print("unique phones:", len(encoder.classes_))
print("acoustic vectors:", len(y_acoustic))
