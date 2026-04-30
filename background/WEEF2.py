import os
import json
import librosa
import numpy as np
import pandas as pd
import kagglehub
from collections import Counter
import random
import pickle
import soundfile as sf
#a

segment_counter = 0

def process_sprsoound_dataset(wav_folders, json_folder_mapping, output_folder, counter):
    for wav_folder in wav_folders:
        print(f"Processing folder: {wav_folder}")

        for root, _, files in os.walk(wav_folder):
            for filename in files:
                if filename.endswith(".wav"):
                    base = os.path.splitext(filename)[0]
                    audio_path = os.path.join(root, filename)

                    json_path = None
                    for json_folder in json_folder_mapping.get(wav_folder, []):
                        candidate = os.path.join(json_folder, base + ".json")
                        if os.path.exists(candidate):
                            json_path = candidate
                            break

                    if json_path:
                        _, counter = slice_audio_with_annotations(
                            audio_path,
                            json_path,
                            output_folder,
                            counter
                        )

    return counter

def save_segment_wav(y_segment, sr, event_type, output_folder, counter):
    os.makedirs(output_folder, exist_ok=True)

    max_val = np.max(np.abs(y_segment)) + 1e-9
    y_segment = y_segment / max_val

    filename = f"segment_{counter:06d}_{event_type}.wav"
    filepath = os.path.join(output_folder, filename)

    sf.write(filepath, y_segment, sr)

    return counter + 1

def slice_audio_with_annotations(audio_filepath, json_filepath, output_folder, counter):
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
    except:
        return [], counter

    try:
        y_full, sr_full = librosa.load(audio_filepath, sr=None)
    except:
        return [], counter

    event_annotations = json_data.get('event_annotation', [])
    sliced_data = []

    for event in event_annotations:
        start_ms = int(event.get('start', 0))
        end_ms = int(event.get('end', 0))
        event_type = event.get('type', 'Unknown')

        start_sample = int(start_ms / 1000 * sr_full)
        end_sample = int(end_ms / 1000 * sr_full)

        if start_sample < end_sample:
            y_segment = y_full[start_sample:end_sample]

            counter = save_segment_wav(y_segment, sr_full, event_type, output_folder, counter)

            sliced_data.append({
                'event_type': event_type
            })

    return sliced_data, counter


def slice_audio_from_txt_annotations(audio_filepath, txt_filepath, output_folder, counter):
    sliced_data = []

    try:
        y_full, sr_full = librosa.load(audio_filepath, sr=None)
    except:
        return [], counter

    try:
        with open(txt_filepath, 'r') as f:
            lines = f.readlines()
    except:
        return [], counter

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            try:
                start_time_s = float(parts[0])
                end_time_s = float(parts[1])
                has_wheeze = int(parts[2])
                has_crackle = int(parts[3])

                event_type = 'Normal'
                if has_wheeze and has_crackle:
                    event_type = 'Wheeze+Crackle'
                elif has_wheeze:
                    event_type = 'Wheeze'
                elif has_crackle:
                    event_type = 'Crackle'

                start_sample = int(start_time_s * sr_full)
                end_sample = int(end_time_s * sr_full)

                if start_sample < end_sample:
                    y_segment = y_full[start_sample:end_sample]

                    # 🔥 SAVE WAV
                    counter = save_segment_wav(y_segment, sr_full, event_type, output_folder, counter)

                    sliced_data.append({
                        'event_type': event_type
                    })

            except:
                pass

    return sliced_data, counter


def augment_audio_segment(segment, sr, max_augmentations=3):
    """Apply random augmentation to audio segment."""
    augmented_segment = segment.copy()
    num_augmentations = random.randint(1, max_augmentations)
    augmentation_types = ['add_noise', 'pitch_shift', 'time_stretch']
    random.shuffle(augmentation_types)

    for i in range(num_augmentations):
        aug_type = augmentation_types[i % len(augmentation_types)]

        if aug_type == 'add_noise':
            noise_amplitude = 0.005 * random.uniform(0.5, 2.0)
            augmented_segment += noise_amplitude * np.random.randn(len(augmented_segment))
        elif aug_type == 'pitch_shift':
            n_steps = random.uniform(-1, 1)
            augmented_segment = librosa.effects.pitch_shift(y=augmented_segment, sr=sr, n_steps=n_steps)
        elif aug_type == 'time_stretch':
            rate = random.uniform(0.9, 1.1)
            if len(augmented_segment) < sr // 5:
                pad_length = sr // 5 - len(augmented_segment)
                augmented_segment = np.pad(augmented_segment, (0, pad_length), 'constant')
            augmented_segment = librosa.effects.time_stretch(y=augmented_segment, rate=rate)

    return augmented_segment



def process_respiratory_database(dataset_path, output_folder, counter):
    processed_data = {}

    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith('.wav'):
                base = os.path.splitext(filename)[0]
                audio_path = os.path.join(root, filename)
                txt_path = os.path.join(root, base + '.txt')

                if os.path.exists(txt_path):
                    segments, counter = slice_audio_from_txt_annotations(
                        audio_path,
                        txt_path,
                        output_folder,
                        counter
                    )

    return counter


def perform_augmentation(segments_list, target_count=2000):
    """Perform data augmentation to balance classes."""
    current_counts = Counter(seg['event_type'] for seg in segments_list)
    classes_to_augment = ['Wheeze', 'Fine Crackle', 'Coarse Crackle', 'Rhonchi', 'Wheeze+Crackle', 'Stridor', 'Crackle']

    segments_by_type = {event_type: [] for event_type in current_counts.keys()}
    for segment_info in segments_list:
        segments_by_type[segment_info['event_type']].append(segment_info)

    augmented_count = 0
    for event_type in classes_to_augment:
        if event_type not in current_counts:
            continue
        current_count = current_counts[event_type]
        needed_samples = target_count - current_count

        if needed_samples > 0:
            print(f"Augmenting {event_type}: Need {needed_samples} samples")
            for _ in range(needed_samples):
                original = random.choice(segments_by_type[event_type])
                augmented_wav = augment_audio_segment(original['segment'], sr=original['sr'])
                segments_list.append({
                    'segment': augmented_wav,
                    'event_type': original['event_type'],
                    'start_ms': original['start_ms'],
                    'end_ms': original['end_ms'],
                    'sr': original['sr']
                })
                augmented_count += 1

    return augmented_count

def save_segments_to_json(segments_list, output_folder="Data_Augmentation2"):
    """Save each segment as an individual JSON file."""
    os.makedirs(output_folder, exist_ok=True)

    for i, seg in enumerate(segments_list):
        json_data = {
            "event_type": seg["event_type"],
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "sr": seg["sr"],
            # Convert numpy array → list for JSON serialization
            "signal": seg["segment"].tolist()
        }

        filename = f"segment_{i:06d}.json"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, "w") as f:
            json.dump(json_data, f)

    print(f"\nSaved {len(segments_list)} JSON files in '{output_folder}'")


def main():
    print("=" * 60)
    print("Starting Audio Dataset Processing")
    print("=" * 60)

    output_folder = "segments_wav_3"
    segment_counter = 0

    # =========================
    # 1. SPRSound
    # =========================
    print("\n[1/2] Processing SPRSound dataset...")

    wav_folders = [
        r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2022\test2022_wav",
        r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2022\train2022_wav",
        r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2023\test2023_wav",
        r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2024\test2024_wav"
    ]

    json_folder_mapping = {
        wav_folders[0]: [
            r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2022\test2022_json\inter_test_json",
            r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2022\test2022_json\intra_test_json"
        ],
        wav_folders[1]: [r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2022\train2022_json"],
        wav_folders[2]: [r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2023\test2023_json"],
        wav_folders[3]: [r"C:\Users\ADMIN\Documents\BREATH-IA\BreathIA\SPRSound\BioCAS2024\test2024_json"]
    }

    segment_counter = process_sprsoound_dataset(wav_folders,json_folder_mapping,output_folder,segment_counter)

    print(f"Segments after SPRSound: {segment_counter}")

    # =========================
    # 2. Respiratory DB
    # =========================
    print("\n[2/2] Processing Respiratory Sound Database...")

    try:
        path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")

        second_dataset_path = os.path.join(
            path,
            'Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
        )

        third_dataset_path = os.path.join(
            path,
            'respiratory_sound_database'
        )

        segment_counter = process_respiratory_database(
            second_dataset_path,
            output_folder,
            segment_counter
        )

        segment_counter = process_respiratory_database(
            third_dataset_path,
            output_folder,
            segment_counter
        )

        print(f"Total segments saved: {segment_counter}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()