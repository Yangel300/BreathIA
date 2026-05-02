import os
import librosa
import random
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import soundfile as sf
def load_segments_from_folder(folder):
    segments_list = []

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)

            try:
                y, sr = librosa.load(filepath, sr=None)

                # Extract event type robustly
                if "Normal" in file:
                    event_type = "Normal"
                elif "wheeze+crackle" in file:
                    event_type = "wheeze+crackle"
                elif "Wheeze" in file:
                    event_type = "Wheeze"
                elif "Crackle" in file:
                    event_type = "Crackle"
                elif "Rhonchi" in file:
                    event_type = "Rhonchi"
                elif "Stridor" in file:
                    event_type = "Stridor"
                else:
                    continue

                segments_list.append({
                    'segment': y,
                    'event_type': event_type,
                    'sr': sr
                })

            except:
                pass

    return segments_list
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
def save_segment_wav(y_segment, sr, event_type, output_folder, counter):
    os.makedirs(output_folder, exist_ok=True)

    max_val = np.max(np.abs(y_segment)) + 1e-9
    y_segment = y_segment / max_val

    filename = f"segment_{counter:06d}_{event_type}.wav"
    filepath = os.path.join(output_folder, filename)

    sf.write(filepath, y_segment, sr)

    return counter + 1
def augment_from_folder(input_folder, output_folder, target_count=2000):
    from collections import Counter
    import os

    segments = load_segments_from_folder(input_folder)
    counts = Counter(s['event_type'] for s in segments)

    os.makedirs(output_folder, exist_ok=True)

    counter = 0
    total_augmented = 0

    for event_type, count in counts.items():
        if count >= target_count:
            continue

        needed = target_count - count
        print(f"{event_type}: need {needed}")

        candidates = [s for s in segments if s['event_type'] == event_type]

        for _ in range(needed):
            original = random.choice(candidates)
            augmented = augment_audio_segment(original['segment'], original['sr'])

            counter = save_segment_wav(
                augmented,
                original['sr'],
                event_type,
                output_folder,   
                counter
            )

            total_augmented += 1

    return total_augmented


def run_augmentation(input_folder, output_folder, target_count=1000):
    print("=" * 60)
    print("Starting Data Augmentation")
    print("=" * 60)

    # Load segments (if needed for metadata)
    segments = load_segments_from_folder(input_folder)

    print(f"Loaded segments: {len(segments)}")

    augmented = augment_from_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        target_count=target_count
    )

    print(f"Augmented samples: {augmented}")

    return augmented

if __name__ == "__main__":
    input_folder = "/home/ares/Documents/BREATH/code/Segments_wav_4"
    output_folder = "/home/ares/Documents/BREATH/code/Augmented_Segments_wav_4"

    run_augmentation(input_folder, output_folder, target_count=1000)
