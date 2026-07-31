import subprocess
from pathlib import Path
import librosa
import soundfile as sf
import re

from textgrid import TextGrid

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).parent.parent

DATASET = ROOT / "dataset"
WAV_DIR = DATASET / "wavs"
META = DATASET / "metadata.csv"

ASSETS = ROOT / "assets"

CORPUS = ASSETS / "corpus"
ALIGNMENTS = ASSETS / "alignments"
LABELS = ASSETS / "labels"

SR = 16000

# =========================
# CREATE DIRS
# =========================

for d in [
    CORPUS,
    ALIGNMENTS,
    LABELS
]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.lower()

    text = text.replace("’", "'")
    text = text.replace("`", "'")

    text = re.sub(r"[^а-щьюяєіїґa-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================
# PREPROCESS
# =========================

def preprocess():

    print("[1] preprocessing...")

    # clean corpus
    for f in CORPUS.glob("*"):
        f.unlink()

    count = 0

    with open(META, encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split("|")

            if len(parts) < 2:
                continue

            file_id = parts[0]
            text = clean_text(parts[1])

            wav_path = WAV_DIR / f"{file_id}.wav"

            if not wav_path.exists():
                continue

            try:

                audio, _ = librosa.load(wav_path, sr=SR)

                out_wav = CORPUS / f"{file_id}.wav"
                out_txt = CORPUS / f"{file_id}.txt"

                sf.write(out_wav, audio, SR)
                out_txt.write_text(text, encoding="utf-8")

                count += 1

            except Exception as e:
                print("ERROR:", wav_path.name, e)

    print(f"[1] corpus files: {count}")

# =========================
# MFA ALIGNMENT
# =========================

def run_mfa():

    print("[2] MFA alignment...")

    # clean old alignments
    for f in ALIGNMENTS.glob("*"):
        f.unlink()

    cmd = f"""
    source ~/miniforge3/etc/profile.d/conda.sh && \
    conda activate mfa && \
    mfa align \
    "{CORPUS}" \
    ukrainian_mfa \
    ukrainian_mfa \
    "{ALIGNMENTS}" \
    --clean \
    --single_speaker
    """

    subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        check=True
    )

# =========================
# TEXTGRID -> HTS LABELS
# =========================

def make_labels():

    print("[3] generating HTS labels...")

    # clean old labels
    for f in LABELS.glob("*"):
        f.unlink()

    tg_files = list(ALIGNMENTS.glob("*.TextGrid"))

    print("TextGrids:", len(tg_files))

    ok = 0

    for tg_path in tg_files:

        try:

            tg = TextGrid.fromFile(str(tg_path))

            out_path = LABELS / f"{tg_path.stem}.lab"

            found = False

            with open(out_path, "w", encoding="utf-8") as f:

                phone_tier = None

                # шукаємо phones tier
                for tier in tg.tiers:

                    name = tier.name.lower()

                    if "phone" in name:
                        phone_tier = tier
                        break

                if phone_tier is None:
                    print("NO PHONE TIER:", tg_path.name)
                    continue

                prev_end = 0

                for interval in phone_tier:

                    label = interval.mark.strip()

                    # пропускаємо шум
                    if label in ["", "sil", "sp", "spn", "<unk>"]:
                        continue

                    start = interval.minTime
                    end = interval.maxTime

                    if end <= start:
                        continue

                    if start - prev_end > 0.05:
                        # HTK units (100ns)
                        start_htk = int(prev_end * 10000000)
                        end_htk = int(start * 10000000)

                        f.write(f"{start_htk} {end_htk} pau\n")

                    prev_end = end

                    # HTK units (100ns)
                    start_htk = int(start * 10000000)
                    end_htk = int(end * 10000000)

                    f.write(f"{start_htk} {end_htk} {label}\n")

                    found = True

            if found:
                ok += 1
            else:
                print("EMPTY:", tg_path.name)

        except Exception as e:
            print("ERROR:", tg_path.name, e)

    print(f"[3] labels done: {ok}/{len(tg_files)}")

# =========================
# MAIN
# =========================

def main():

    preprocess()

    run_mfa()

    make_labels()

    print("\nDONE")
    print("corpus")
    print("alignments")
    print("labels")

if __name__ == "__main__":
    main()
