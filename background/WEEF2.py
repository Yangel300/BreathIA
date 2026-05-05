import os
import sys
import json
import librosa
import numpy as np
import kagglehub
import random
import soundfile as sf
from collections import Counter

# =========================================
# PATH BASE (clave para evitar errores)
# =========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(CURRENT_DIR, "datasets")
print(CURRENT_DIR)
os.makedirs(DATASET_BASE, exist_ok=True)


# =========================================
# UTILIDADES
# =========================================
def ensure_dir(path):
    if not os.path.exists(path):
        print(f"[INFO] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)


# =========================================
# AUDIO HELPERS
# =========================================
def hann_fade(audio, sr, fade_ms=20):
    fade_samples = int(sr * fade_ms / 1000)

    if len(audio) < 2 * fade_samples:
        return audio

    hann = np.hanning(2 * fade_samples)
    audio[:fade_samples] *= hann[:fade_samples]
    audio[-fade_samples:] *= hann[fade_samples:]

    return audio


def save_segment_wav(y_segment, sr, event_type, output_folder, counter):
    ensure_dir(output_folder)

    y_segment = hann_fade(y_segment, sr)

    max_val = np.max(np.abs(y_segment)) + 1e-9
    y_segment = y_segment / max_val

    filename = f"segment_{counter:06d}_{event_type}.wav"
    filepath = os.path.join(output_folder, filename)

    sf.write(filepath, y_segment, sr)

    return counter + 1


def enforce_min_duration(y_full, start_sample, end_sample, sr, min_duration=0.8):
    min_samples = int(min_duration * sr)
    current_len = end_sample - start_sample

    if current_len >= min_samples:
        return start_sample, end_sample

    center = (start_sample + end_sample) // 2
    half = min_samples // 2

    new_start = max(0, center - half)
    new_end = min(len(y_full), center + half)

    if (new_end - new_start) < min_samples:
        return None, None

    return new_start, new_end


# =========================================
# SLICING JSON
# =========================================
def slice_audio_with_annotations(audio_filepath, json_filepath, output_folder, counter):
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"[WARN] JSON error: {json_filepath} -> {e}")
        return [], counter

    try:
        y_full, sr_full = librosa.load(audio_filepath, sr=None)
    except Exception as e:
        print(f"[WARN] Audio load error: {audio_filepath} -> {e}")
        return [], counter

    sliced_data = []

    for event in json_data.get('event_annotation', []):
        start_ms = int(event.get('start', 0))
        end_ms = int(event.get('end', 0))
        raw_type = event.get('type', 'Unknown').lower()

        if "wheeze" in raw_type:
            event_type = "Wheeze"
        elif "crackle" in raw_type:
            event_type = "Crackle"
        elif "normal" in raw_type:
            event_type = "Normal"
        else:
            continue

        start = int(start_ms / 1000 * sr_full)
        end = int(end_ms / 1000 * sr_full)

        if start < end:
            start, end = enforce_min_duration(y_full, start, end, sr_full)

            if start is None:
                continue

            y_segment = y_full[start:end]

            counter = save_segment_wav(
                y_segment, sr_full, event_type, output_folder, counter
            )

            sliced_data.append({'event_type': event_type})

    return sliced_data, counter


# =========================================
# DATASET SPRSOUND
# =========================================
def process_sprsound_dataset(basepath, output_folder, counter):
    print(f"[INFO] Using SPRSound basepath: {basepath}")

    wav_folders = [
        os.path.join(basepath, "BioCAS2022/test2022_wav"),
        os.path.join(basepath, "BioCAS2022/train2022_wav"),
        os.path.join(basepath, "BioCAS2023/test2023_wav"),
        os.path.join(basepath, "BioCAS2024/test2024_wav"),
    ]

    validation_wav_folder = os.path.join(basepath, "BioCAS2025/test2025_wav")

    json_mapping = {
        wav_folders[0]: [
            os.path.join(basepath, "BioCAS2022/test2022_json/inter_test_json"),
            os.path.join(basepath, "BioCAS2022/test2022_json/intra_test_json"),
        ],
        wav_folders[1]: [os.path.join(basepath, "BioCAS2022/train2022_json")],
        wav_folders[2]: [os.path.join(basepath, "BioCAS2023/test2023_json")],
        wav_folders[3]: [os.path.join(basepath, "BioCAS2024/test2024_json")],

        validation_wav_folder: [os.path.join(basepath, "BioCAS2025/test2025_json")],
    }

    for wav_folder in wav_folders:
        if not os.path.exists(wav_folder):
            print(f"[WARN] Missing folder: {wav_folder}")
            continue

        for root, _, files in os.walk(wav_folder):
            for file in files:
                if file.endswith(".wav"):
                    base = os.path.splitext(file)[0]
                    audio_path = os.path.join(root, file)

                    json_path = None
                    for jf in json_mapping.get(wav_folder, []):
                        candidate = os.path.join(jf, base + ".json")
                        if os.path.exists(candidate):
                            json_path = candidate
                            break

                    if json_path:
                        _, counter = slice_audio_with_annotations(
                            audio_path, json_path, output_folder, counter
                        )
    validation_output = os.path.join(os.path.dirname(output_folder),"Validation_Segments_wav")
    ensure_dir(validation_output)

    if os.path.exists(validation_wav_folder):
        print(f"[INFO] Processing validation set: {validation_wav_folder}")

        for root, _, files in os.walk(validation_wav_folder):
            for file in files:
                if file.endswith(".wav"):
                    base = os.path.splitext(file)[0]
                    audio_path = os.path.join(root, file)

                    json_path = None
                    for jf in json_mapping.get(validation_wav_folder, []):
                        candidate = os.path.join(jf, base + ".json")
                        if os.path.exists(candidate):
                            json_path = candidate
                            break

                    if json_path:
                        _, counter = slice_audio_with_annotations(
                            audio_path, json_path, validation_output, counter
                        )
    else:
        print(f"[WARN] Missing validation folder: {validation_wav_folder}")

    return counter


# =========================================
# RESPIRATORY DATASET
# =========================================
def process_respiratory_database(dataset_path, output_folder, counter):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                base = os.path.splitext(file)[0]
                audio = os.path.join(root, file)
                txt = os.path.join(root, base + ".txt")

                if os.path.exists(txt):
                    _, counter = slice_audio_from_txt_annotations(
                        audio, txt, output_folder, counter
                    )

    return counter


def slice_audio_from_txt_annotations(audio_filepath, txt_filepath, output_folder, counter):
    try:
        y_full, sr_full = librosa.load(audio_filepath, sr=None)
    except Exception as e:
        print(f"[WARN] Audio error: {e}")
        return [], counter

    try:
        with open(txt_filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[WARN] TXT error: {e}")
        return [], counter

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue

        try:
            start = float(parts[0])
            end = float(parts[1])
            wheeze = int(parts[2])
            crackle = int(parts[3])

            if wheeze and crackle:
                continue
            elif wheeze:
                event = "Wheeze"
            elif crackle:
                event = "Crackle"
            else:
                event = "Normal"

            s = int(start * sr_full)
            e = int(end * sr_full)

            s, e = enforce_min_duration(y_full, s, e, sr_full)
            if s is None:
                continue

            counter = save_segment_wav(
                y_full[s:e], sr_full, event, output_folder, counter
            )

        except Exception:
            continue

    return [], counter


# =========================================
# MAIN PIPELINE
# =========================================
def run_slicing(output_folder, sprsound_path=None):
    print("=" * 60)
    print("STARTING DATASET PIPELINE")
    print("=" * 60)

    ensure_dir(output_folder)

    counter = 0

    # -------------------------
    # SPRSound
    # -------------------------
    if sprsound_path is None:
        sprsound_path = os.path.join(DATASET_BASE, "SPRSound")

    if os.path.exists(sprsound_path):
        counter = process_sprsound_dataset(
            sprsound_path,
            output_folder,
            counter
        )
    else:
        print(f"[WARN] SPRSound not found at {sprsound_path}")

    print(f"[INFO] Segments after SPRSound: {counter}")

    # -------------------------
    # Respiratory DB (Kaggle)
    # -------------------------
    print("[INFO] Downloading Respiratory DB...")
    path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")

    dataset_path = os.path.join(
        path,
        "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
    )

    counter = process_respiratory_database(
        dataset_path,
        output_folder,
        counter
    )

    print(f"[DONE] Total segments: {counter}")

    return counter
# =========================================
# ENTRYPOINT
# =========================================
if __name__ == "__main__":
    OUTPUT = os.path.join(CURRENT_DIR, "Segments_wav_4")
    run_slicing(OUTPUT)

