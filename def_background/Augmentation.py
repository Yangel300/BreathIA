import os
import random
import numpy as np
import soundfile as sf

from scipy.signal import (
    resample,
    butter,
    lfilter
)

from collections import Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_FOLDER = (
    r"C:\Users\Oficina 01\Documents\Breath\BreathIA\raw_dataset_with_rules"
)

OUTPUT_FOLDER = (
    r"C:\Users\Oficina 01\Documents\Breath\BreathIA\augmentation_dataset_with_rules"
)

TARGET_SR = 22050

# ---------------------------------------------------------
# TARGET COUNTS
# ---------------------------------------------------------

TARGET_COUNTS = {
    "Normal": 4200,
    "Wheeze": 2200,
    "Fine_Crackle": 3000
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# LOAD DATASET
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

            new_length = int(
                len(audio) * TARGET_SR / sr
            )

            audio = resample(audio, new_length)

            sr = TARGET_SR

        # -------------------------------------------------
        # LABELS
        # -------------------------------------------------

        name = file.lower()

        if "normal" in name:
            label = "Normal"

        elif "wheeze" in name:
            label = "Wheeze"

        elif "fine_crackle" in name:
            label = "Fine_Crackle"

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

print(f"\nLoaded files: {len(segments)}")

# =========================================================
# COUNTS
# =========================================================

counts = Counter([s["label"] for s in segments])

print("\nOriginal counts:\n")

for k, v in counts.items():
    print(f"{k}: {v}")

# =========================================================
# SAVE ORIGINALS WITH UNDERSAMPLING
# =========================================================

print("\nSaving balanced originals...\n")

saved_counter = 1

balanced_segments = []

for label in TARGET_COUNTS.keys():

    class_segments = [
        s for s in segments
        if s["label"] == label
    ]

    # -----------------------------------------------------
    # UNDERSAMPLING NORMAL
    # -----------------------------------------------------

    if len(class_segments) > TARGET_COUNTS[label]:

        class_segments = random.sample(
            class_segments,
            TARGET_COUNTS[label]
        )

    balanced_segments.extend(class_segments)

# Save originals
for seg in balanced_segments:

    filename = (
        f"sample_{saved_counter:06d}_{seg['label']}.wav"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    y = seg["audio"]

    y = y / (np.max(np.abs(y)) + 1e-9)

    sf.write(
        output_path,
        y,
        TARGET_SR
    )

    saved_counter += 1

# =========================================================
# AUGMENTATIONS
# =========================================================

def add_noise(y):

    noise_amp = random.uniform(
        0.0002,
        0.002
    )

    noise = noise_amp * np.random.randn(len(y))

    return y + noise


def gain(y):

    g = random.uniform(
        0.9,
        1.1
    )

    return y * g


def stretch(y):

    rate = random.uniform(
        0.97,
        1.03
    )

    new_length = int(len(y) / rate)

    y2 = resample(y, new_length)

    target = TARGET_SR * 3

    if len(y2) > target:
        y2 = y2[:target]

    else:
        y2 = np.pad(
            y2,
            (0, target - len(y2))
        )

    return y2


# ---------------------------------------------------------
# BANDPASS PERTURBATION
# ---------------------------------------------------------

def bandpass_filter(y):

    low = random.uniform(80, 150)
    high = random.uniform(1200, 2200)

    nyquist = TARGET_SR / 2

    low = low / nyquist
    high = high / nyquist

    b, a = butter(
        4,
        [low, high],
        btype='band'
    )

    return lfilter(b, a, y)


# ---------------------------------------------------------
# FREQUENCY MASKING
# ---------------------------------------------------------

def frequency_mask(y):

    Y = np.fft.rfft(y)

    n = len(Y)

    start = random.randint(
        0,
        int(n * 0.7)
    )

    width = random.randint(
        int(n * 0.01),
        int(n * 0.05)
    )

    Y[start:start+width] = 0

    return np.fft.irfft(Y)


# ---------------------------------------------------------
# TIME MASKING
# ---------------------------------------------------------

def time_mask(y):

    y = y.copy()

    length = len(y)

    mask_size = random.randint(
        int(length * 0.01),
        int(length * 0.05)
    )

    start = random.randint(
        0,
        length - mask_size
    )

    y[start:start+mask_size] = 0

    return y


# ---------------------------------------------------------
# RESPIRATORY ENVELOPE MODULATION
# ---------------------------------------------------------

def respiratory_modulation(y):

    t = np.linspace(
        0,
        1,
        len(y)
    )

    freq = random.uniform(
        0.1,
        0.4
    )

    envelope = (
        0.75 +
        0.25 * np.sin(
            2 * np.pi * freq * t
        )
    )

    return y * envelope


# =========================================================
# MAIN AUGMENT FUNCTION
# =========================================================

AUGMENTS = [
    add_noise,
    gain,
    stretch,
    bandpass_filter,
    frequency_mask,
    time_mask,
    respiratory_modulation
]

def augment(y):

    y_aug = y.copy()

    n_aug = random.randint(2, 4)

    chosen = random.sample(
        AUGMENTS,
        n_aug
    )

    for aug in chosen:

        try:
            y_aug = aug(y_aug)

        except Exception:
            pass

    # Normalize
    y_aug = y_aug / (
        np.max(np.abs(y_aug)) + 1e-9
    )

    return y_aug


# =========================================================
# AUGMENT MINORITY CLASSES
# =========================================================

print("\nStarting augmentation...\n")

balanced_counts = Counter([
    s["label"] for s in balanced_segments
])

for label, target in TARGET_COUNTS.items():

    current = balanced_counts[label]

    print("=" * 50)
    print(label)
    print(f"Current: {current}")
    print(f"Target:  {target}")

    if current >= target:

        print("Already balanced")
        continue

    needed = target - current

    print(f"Generating: {needed}")

    candidates = [
        s for s in balanced_segments
        if s["label"] == label
    ]

    for i in range(needed):

        original = random.choice(candidates)

        augmented = augment(
            original["audio"]
        )

        filename = (
            f"sample_{saved_counter:06d}_{label}.wav"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        sf.write(
            output_path,
            augmented,
            TARGET_SR
        )

        saved_counter += 1

        if i % 100 == 0:

            print(
                f"{label}: "
                f"{i}/{needed}"
            )

# =========================================================
# FINAL COUNTS
# =========================================================

print("\nDONE\n")

final_counter = Counter()

for file in os.listdir(OUTPUT_FOLDER):

    name = file.lower()

    if "normal" in name:
        final_counter["Normal"] += 1

    elif "wheeze" in name:
        final_counter["Wheeze"] += 1

    elif "fine_crackle" in name:
        final_counter["Fine_Crackle"] += 1

print("Final counts:\n")

for k, v in final_counter.items():
    print(f"{k}: {v}")