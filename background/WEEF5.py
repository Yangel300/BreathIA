import os
import sys
import librosa
import random
import numpy as np
import soundfile as sf
from collections import Counter

# =========================================
# PATH BASE (evita problemas de rutas)
# =========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================
# UTILIDADES
# =========================================
def ensure_dir(path):
    if not os.path.exists(path):
        print(f"[INFO] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)


# =========================================
# LOAD SEGMENTS
# =========================================
def load_segments_from_folder(folder):
    segments_list = []

    if not os.path.exists(folder):
        print(f"[ERROR] Folder does not exist: {folder}")
        return segments_list

    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue

        filepath = os.path.join(folder, file)

        try:
            y, sr = librosa.load(filepath, sr=None)

            # Clasificación robusta
            name = file.lower()
            if "normal" in name:
                event_type = "Normal"
            elif "wheeze" in name:
                event_type = "Wheeze"
            elif "crackle" in name:
                event_type = "Crackle"
            else:
                continue

            segments_list.append({
                'segment': y,
                'event_type': event_type,
                'sr': sr
            })

        except Exception as e:
            print(f"[WARN] Failed loading {file}: {e}")

    return segments_list


# =========================================
# AUGMENTATION
# =========================================
def augment_audio_segment(segment, sr, max_augmentations=3):
    augmented = segment.copy()

    num_aug = random.randint(1, max_augmentations)
    aug_types = ['add_noise', 'pitch_shift', 'time_stretch']
    random.shuffle(aug_types)

    for i in range(num_aug):
        aug = aug_types[i % len(aug_types)]

        try:
            if aug == 'add_noise':
                noise_amp = 0.005 * random.uniform(0.5, 2.0)
                augmented += noise_amp * np.random.randn(len(augmented))

            elif aug == 'pitch_shift':
                steps = random.uniform(-1, 1)
                augmented = librosa.effects.pitch_shift(
                    y=augmented, sr=sr, n_steps=steps
                )

            elif aug == 'time_stretch':
                rate = random.uniform(0.9, 1.1)

                if len(augmented) < sr // 5:
                    pad = sr // 5 - len(augmented)
                    augmented = np.pad(augmented, (0, pad))

                augmented = librosa.effects.time_stretch(
                    y=augmented, rate=rate
                )

        except Exception as e:
            print(f"[WARN] Augmentation error: {e}")

    return augmented


# =========================================
# SAVE
# =========================================
def save_segment_wav(y_segment, sr, event_type, output_folder, counter):
    ensure_dir(output_folder)

    # Normalización segura
    max_val = np.max(np.abs(y_segment)) + 1e-9
    y_segment = y_segment / max_val

    filename = f"segment_{counter:06d}_{event_type}.wav"
    filepath = os.path.join(output_folder, filename)

    try:
        sf.write(filepath, y_segment, sr)
    except Exception as e:
        print(f"[ERROR] Failed saving {filepath}: {e}")
        return counter

    return counter + 1


# =========================================
# AUGMENT FROM FOLDER
# =========================================
def augment_from_folder(input_folder, output_folder, target_count):
    segments = load_segments_from_folder(input_folder)

    if len(segments) == 0:
        print("[ERROR] No segments loaded. Abort.")
        return 0

    counts = Counter(s['event_type'] for s in segments)
    print(f"[INFO] Current counts: {dict(counts)}")

    ensure_dir(output_folder)

    counter = 0
    total_augmented = 0

    for event_type, count in counts.items():
        print(f"[INFO] Processing {event_type}: current {count}, target {target_count}")
        if count >= target_count:
            continue

        needed = target_count - count
        print(f"[INFO] {event_type}: need {needed}")

        candidates = [s for s in segments if s['event_type'] == event_type]

        if len(candidates) == 0:
            continue

        for _ in range(needed):
            original = random.choice(candidates)

            augmented = augment_audio_segment(
                original['segment'],
                original['sr']
            )

            counter = save_segment_wav(
                augmented,
                original['sr'],
                event_type,
                output_folder,
                counter
            )

            total_augmented += 1

    return total_augmented


# =========================================
# MAIN PIPELINE
# =========================================
def run_augmentation(input_folder, output_folder, target_count):
    print("=" * 60)
    print("STARTING DATA AUGMENTATION")
    print("=" * 60)

    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    print(f"[INFO] Input:  {input_folder}")
    print(f"[INFO] Output: {output_folder}")

    segments = load_segments_from_folder(input_folder)
    print(f"[INFO] Loaded segments: {len(segments)}")

    augmented = augment_from_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        target_count=target_count
    )

    print(f"[DONE] Augmented samples: {augmented}")

    return augmented


# =========================================
# ENTRYPOINT
# =========================================
if __name__ == "__main__":
    INPUT = os.path.join(CURRENT_DIR, "Segments_wav_4")
    OUTPUT = os.path.join(CURRENT_DIR, "Augmented_Segments_wav_4")

    run_augmentation(INPUT, OUTPUT, target_count=4000)