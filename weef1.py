import os
import json
import librosa
import numpy as np
import pandas as pd
import kagglehub
from collections import Counter
import random
import pickle


def slice_audio_with_annotations(audio_filepath, json_filepath):
    """Slices an audio file into segments based on event annotations from a JSON file."""
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON {json_filepath}: {e}")
        return []

    try:
        y_full, sr_full = librosa.load(audio_filepath)
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading audio {audio_filepath}: {e}")
        return []

    event_annotations = json_data.get('event_annotation', [])
    sliced_data = []

    for event in event_annotations:
        start_ms = int(event.get('start', 0))
        end_ms = int(event.get('end', 0))
        event_type = event.get('type', 'Unknown')

        start_sample = max(0, int(start_ms / 1000.0 * sr_full))
        end_sample = min(len(y_full), int(end_ms / 1000.0 * sr_full))

        if start_sample < end_sample:
            y_segment = y_full[start_sample:end_sample]
            sliced_data.append({
                'segment': y_segment,
                'event_type': event_type,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'sr': sr_full
            })
    return sliced_data


def slice_audio_from_txt_annotations(audio_filepath, txt_filepath):
    """Slices an audio file into segments based on event annotations from a TXT file."""
    sliced_data = []

    try:
        y_full, sr_full = librosa.load(audio_filepath)
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading audio {audio_filepath}: {e}")
        return []

    try:
        with open(txt_filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"TXT file not found: {txt_filepath}")
        return []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            try:
                start_time_s = float(parts[0])
                end_time_s = float(parts[1])
                has_wheeze = int(parts[2])
                has_crackle = int(parts[3])

                event_type = 'Normal'
                if has_wheeze == 1 and has_crackle == 1:
                    event_type = 'Wheeze+Crackle'
                elif has_wheeze == 1:
                    event_type = 'Wheeze'
                elif has_crackle == 1:
                    event_type = 'Crackle'

                start_sample = max(0, int(start_time_s * sr_full))
                end_sample = min(len(y_full), int(end_time_s * sr_full))

                if start_sample < end_sample:
                    y_segment = y_full[start_sample:end_sample]
                    sliced_data.append({
                        'segment': y_segment,
                        'event_type': event_type,
                        'start_ms': int(start_time_s * 1000),
                        'end_ms': int(end_time_s * 1000),
                        'sr': sr_full
                    })
            except ValueError:
                pass
    return sliced_data


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


def process_sprsoound_dataset(wav_folders, json_folder_mapping):
    """Process SPRSound dataset."""
    processed_data = {}
    for wav_folder in wav_folders:
        print(f"Processing folder: {wav_folder}")
        for root, _, files in os.walk(wav_folder):
            for filename in files:
                if filename.endswith(".wav"):
                    audio_base_filename = os.path.splitext(filename)[0]
                    audio_filepath = os.path.join(root, filename)
                    json_filepath = None

                    for json_candidate_folder in json_folder_mapping.get(wav_folder, []):
                        candidate_json_path = os.path.join(json_candidate_folder, audio_base_filename + ".json")
                        if os.path.exists(candidate_json_path):
                            json_filepath = candidate_json_path
                            break

                    if json_filepath:
                        segments = slice_audio_with_annotations(audio_filepath, json_filepath)
                        if segments:
                            processed_data[audio_base_filename] = segments
    return processed_data


def process_respiratory_database(dataset_path):
    """Process Respiratory Sound Database."""
    processed_data = {}
    print(f"Processing dataset: {dataset_path}")
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith('.wav'):
                audio_base_filename = os.path.splitext(filename)[0]
                audio_filepath = os.path.join(root, filename)
                txt_filepath = os.path.join(root, audio_base_filename + '.txt')

                if os.path.exists(txt_filepath):
                    segments = slice_audio_from_txt_annotations(audio_filepath, txt_filepath)
                    if segments:
                        processed_data[audio_base_filename] = segments
    return processed_data


def perform_augmentation(segments_list, target_count=4000):
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


def main():
    print("=" * 60)
    print("Starting Audio Dataset Processing")
    print("=" * 60)

    # Process SPRSound dataset
    print("\n[1/3] Processing SPRSound dataset...")
    wav_folders = [
        "SPRSound/BioCAS2022/test2022_wav",
        "SPRSound/BioCAS2022/train2022_wav",
        "SPRSound/BioCAS2023/test2023_wav",
        "SPRSound/BioCAS2024/test2024_wav"
    ]

    json_folder_mapping = {
        "SPRSound/BioCAS2022/test2022_wav": [
            "SPRSound/BioCAS2022/test2022_json/inter_test_json",
            "SPRSound/BioCAS2022/test2022_json/intra_test_json"
        ],
        "SPRSound/BioCAS2022/train2022_wav": ["SPRSound/BioCAS2022/train2022_json"],
        "SPRSound/BioCAS2023/test2023_wav": ["SPRSound/BioCAS2023/test2023_json"],
        "SPRSound/BioCAS2024/test2024_wav": ["SPRSound/BioCAS2024/test2024_json"]
    }

    processed_audio_data = process_sprsoound_dataset(wav_folders, json_folder_mapping)
    print(f"SPRSound: {len(processed_audio_data)} audio files processed")

    # Process Respiratory Sound Database
    print("\n[2/3] Processing Respiratory Sound Database...")
    try:
        path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
        second_dataset_path = os.path.join(path, 'Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files')
        third_dataset_path = os.path.join(path, 'respiratory_sound_database')

        second_data = process_respiratory_database(second_dataset_path)
        processed_audio_data.update(second_data)
        print(f"Second dataset: {len(second_data)} audio files processed")

        third_data = process_respiratory_database(third_dataset_path)
        processed_audio_data.update(third_data)
        print(f"Third dataset: {len(third_data)} audio files processed")
    except Exception as e:
        print(f"Warning: Could not process Respiratory Sound Database: {e}")

    # Convert to flat list and analyze
    print("\n[3/3] Data Augmentation...")
    all_segments = []
    for audio_file, segments in processed_audio_data.items():
        all_segments.extend(segments)

    original_counts = Counter(seg['event_type'] for seg in all_segments)
    print("\nOriginal Event Counts:")
    for event_type, count in original_counts.most_common():
        print(f"  - {event_type}: {count}")

    # Perform augmentation
    augmented_count = perform_augmentation(all_segments, target_count=4000)
    print(f"\nTotal augmented segments added: {augmented_count}")

    # Final counts
    final_counts = Counter(seg['event_type'] for seg in all_segments)
    print("\nFinal Event Counts After Augmentation:")
    for event_type, count in final_counts.most_common():
        print(f"  - {event_type}: {count}")

    # Save processed data
    output_file = "processed_audio_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_segments, f)
    print(f"\nProcessed data saved to: {output_file}")
    print(f"Total segments: {len(all_segments)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
