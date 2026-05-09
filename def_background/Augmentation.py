import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import resample
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_FOLDER = r"C:\Users\Oficina 01\Documents\Breath\BreathIA\raw_dataset_with_rules"

TARGET_COUNT = 4000

TARGET_SR = 22050

# =========================================================
# LOAD FILES
# =========================================================

segments = []

print("\nLoading dataset...\n")

for file in os.listdir(INPUT_FOLDER):

    if not file.endswith(".wav"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)

    try:

        audio, sr = sf.read(filepath)

        # Stereo -> mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        if sr != TARGET_SR:

            new_length = int(len(audio) * TARGET_SR / sr)

            audio = resample(audio, new_length)

            sr = TARGET_SR

        # -------------------------------------------------
        # LABEL
        # -------------------------------------------------

        name = file.lower()

        if "normal" in name:
            label = "Normal"

        elif "wheeze" in name:
            label = "Wheeze"

        elif "fine_crackle" in name:
            label = "Fine_Crackle"

        elif "coarse_crackle" in name:
            label = "Coarse_Crackle"

        elif "rhonchi" in name:
            label = "Rhonchi"

        elif "stridor" in name:
            label = "Stridor"

        else:
            continue

        segments.append({
            "audio": audio,
            "sr": sr,
            "label": label
        })

    except Exception as e:

        print(f"ERROR loading {file}")
        print(e)

print(f"\nLoaded: {len(segments)} files")

# =========================================================
# COUNTS
# =========================================================

counts = Counter([s["label"] for s in segments])

print("\nCounts per class:\n")

for k, v in counts.items():
    print(f"{k}: {v}")

# =========================================================
# AUGMENTATION
# =========================================================

def augment(audio):

    y = audio.copy()

    aug_type = random.choice([
        "noise",
        "gain",
        "stretch"
    ])

    # -----------------------------------------------------
    # VERY LIGHT NOISE
    # -----------------------------------------------------

    if aug_type == "noise":

        noise_amp = random.uniform(
            0.0002,
            0.002
        )

        noise = noise_amp * np.random.randn(len(y))

        y = y + noise

    # -----------------------------------------------------
    # LIGHT GAIN
    # -----------------------------------------------------

    elif aug_type == "gain":

        gain = random.uniform(
            0.9,
            1.1
        )

        y = y * gain

    # -----------------------------------------------------
    # VERY LIGHT STRETCH
    # -----------------------------------------------------

    elif aug_type == "stretch":

        rate = random.uniform(
            0.97,
            1.03
        )

        new_length = int(len(y) / rate)

        y = resample(y, new_length)

        target_length = TARGET_SR * 3

        if len(y) > target_length:

            y = y[:target_length]

        else:

            pad = target_length - len(y)

            y = np.pad(y, (0, pad))

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)

    return y

# =========================================================
# SAVE AUGMENTED
# =========================================================

print("\nStarting augmentation...\n")

counter = len(os.listdir(INPUT_FOLDER)) + 1

for label, current_count in counts.items():

    print("=" * 50)
    print(f"{label}")
    print(f"Current: {current_count}")

    if current_count >= TARGET_COUNT:

        print("Skipping")
        continue

    needed = TARGET_COUNT - current_count

    print(f"Generating: {needed}")

    candidates = [
        s for s in segments
        if s["label"] == label
    ]

    for i in range(needed):

        sample = random.choice(candidates)

        augmented = augment(sample["audio"])

        filename = (
            f"sample_{counter:06d}_{label}.wav"
        )

        output_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        sf.write(
            output_path,
            augmented,
            TARGET_SR
        )

        counter += 1

        if i % 100 == 0:

            print(
                f"{label}: "
                f"{i}/{needed}"
            )

print("\nDONE")