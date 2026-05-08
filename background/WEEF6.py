import os
import random
from collections import defaultdict
from pydub import AudioSegment

# Folders containing audio segments
input_folders = [
    r"C:\Users\ADMIN\Documents\BreathIA\Segments_wav_4",
    r"C:\Users\ADMIN\Documents\BreathIA\Validation_Segments_wav"
]

# Output folder
output_folder = r"C:/Users/ADMIN/Documents/BreathIA/Concatenate"

# Desired final duration
segment_length_sec = 3
TARGET_DURATION_MS = segment_length_sec * 1000

# Audio extension
AUDIO_EXTENSION = ".wav"


os.makedirs(output_folder, exist_ok=True)



files_by_symptom = defaultdict(list)

for folder in input_folders:

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    for filename in os.listdir(folder):

        if not filename.lower().endswith(AUDIO_EXTENSION):
            continue

        filepath = os.path.join(folder, filename)

        # Extract symptom from filename
        # Example:
        # segment_000003_Wheeze.wav -> Wheeze

        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("_")

        if len(parts) < 3:
            print(f"Invalid filename format: {filename}")
            continue

        symptom = "_".join(parts[2:])

        files_by_symptom[symptom].append(filepath)



output_counter = 0

for symptom, file_list in files_by_symptom.items():

    print(f"\nProcessing symptom: {symptom}")
    print(f"Total files: {len(file_list)}")

    # Shuffle files randomly
    random.shuffle(file_list)

    current_audio = AudioSegment.empty()

    while len(file_list) > 0:

        filepath = file_list.pop()

        try:
            audio = AudioSegment.from_file(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        # Add audio
        current_audio += audio

        # If reached target duration
        if len(current_audio) >= TARGET_DURATION_MS:

            # Trim exactly to 3 seconds
            final_audio = current_audio[:TARGET_DURATION_MS]

            # Save output
            output_filename = (
                f"concat_{output_counter:06d}_{symptom}.wav"
            )

            output_path = os.path.join(output_folder, output_filename)

            final_audio.export(output_path, format="wav")

            print(f"Saved: {output_filename}")

            output_counter += 1

            # Keep remaining audio after 3 seconds
            current_audio = current_audio[TARGET_DURATION_MS:]

    # Save remaining audio if not empty
    if len(current_audio) > 0:

        # Pad with silence if shorter than 3 seconds
        if len(current_audio) < TARGET_DURATION_MS:
            silence_needed = TARGET_DURATION_MS - len(current_audio)
            current_audio += AudioSegment.silent(duration=silence_needed)

        output_filename = (
            f"concat_{output_counter:06d}_{symptom}.wav"
        )

        output_path = os.path.join(output_folder, output_filename)

        current_audio.export(output_path, format="wav")

        print(f"Saved remaining audio: {output_filename}")

        output_counter += 1

print("\nFinished generating concatenated 3-second files.")

