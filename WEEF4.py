
# --- Code cell ---


# --- Code cell ---
import os
import json
import librosa
import numpy as np
from collections import Counter


def slice_audio_with_annotations(audio_filepath, json_filepath):
  """
  Slices an audio file into segments based on event annotations from a JSON file.

  Args:
    audio_filepath (str): Path to the audio (.wav) file.
    json_filepath (str): Path to the corresponding JSON annotation file.

  Returns:
    list: A list of dictionaries, where each dictionary contains:
          - 'segment': The numpy array of the sliced audio segment.
          - 'event_type': The type of the event (e.g., 'Wheeze').
          - 'start_ms': Start time of the event in milliseconds.
          - 'end_ms': End time of the event in milliseconds.
  """
  try:
    with open(json_filepath, 'r') as f:
      json_data = json.load(f)
  except FileNotFoundError:
    print(f"JSON file not found: {json_filepath}")
    return []
  except json.JSONDecodeError:
    print(f"Error decoding JSON from file: {json_filepath}")
    return []

  try:
    y_full, sr_full = librosa.load(audio_filepath)
  except FileNotFoundError:
    print(f"Audio file not found: {audio_filepath}")
    return []
  except Exception as e:
    print(f"Error loading audio file {audio_filepath}: {e}")
    return []

  event_annotations = json_data.get('event_annotation', [])
  sliced_data = []

  for event in event_annotations:
    start_ms = int(event.get('start', 0))
    end_ms = int(event.get('end', 0))
    event_type = event.get('type', 'Unknown')

    start_time_s = start_ms / 1000.0
    end_time_s = end_ms / 1000.0

    start_sample = int(start_time_s * sr_full)
    end_sample = int(end_time_s * sr_full)

    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(y_full), end_sample)

    if start_sample < end_sample:
      y_segment = y_full[start_sample:end_sample]
      sliced_data.append({
          'segment': y_segment,
          'event_type': event_type,
          'start_ms': start_ms,
          'end_ms': end_ms
      })
  return sliced_data

# Define all relevant folders using existing variables from the notebook state
wavtrainfolder1="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2022/test2022_wav"
wavtrainfolder2="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2022/train2022_wav"
wavtrainfolder3="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2023/test2023_wav"
wavtrainfolder4="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2024/test2024_wav"

wav_folders = [
    wavtrainfolder1,
    wavtrainfolder2,
    wavtrainfolder3,
    wavtrainfolder4
]

jsontrainfolder1="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2022/test2022_json/inter_test_json"
jsontrainfolder2="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2022/test2022_json/intra_test_json"
jsontrainfolder3="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2022/train2022_json"
jsontrainfolder4="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2023/test2023_json"
jsontrainfolder5="/home/ares/Documents/BREATH/code/SPRSound/BioCAS2024/test2024_json"
json_folder_mapping = {
    wavtrainfolder1: [jsontrainfolder1, jsontrainfolder2], # BioCAS2022/test2022_wav maps to inter_test_json and intra_test_json
    wavtrainfolder2: [jsontrainfolder3], # BioCAS2022/train2022_wav maps to train2022_json
    wavtrainfolder3: [jsontrainfolder4], # BioCAS2023/test2023_wav maps to test2023_json
    wavtrainfolder4: [jsontrainfolder5]  # BioCAS2024/test2024_wav maps to test2024_json
}

processed_audio_data = {}

# Iterate through each WAV folder
for wav_folder in wav_folders:
    print(f"Processing folder: {wav_folder}")
    for root, _, files in os.walk(wav_folder):
        for filename in files:
            if filename.endswith(".wav"):
                audio_base_filename = os.path.splitext(filename)[0]
                audio_filepath = os.path.join(root, filename)

                json_filepath = None
                # Check all potential JSON folders for the current wav_folder
                for json_candidate_folder in json_folder_mapping.get(wav_folder, []):
                    candidate_json_path = os.path.join(json_candidate_folder, audio_base_filename + ".json")
                    if os.path.exists(candidate_json_path):
                        json_filepath = candidate_json_path
                        break # Found the JSON file, no need to check other subfolders for this audio file

                if json_filepath:
                    print(f"  Slicing {filename} with annotation from {os.path.basename(json_filepath)}")
                    segments_with_annotations = slice_audio_with_annotations(audio_filepath, json_filepath)
                    if segments_with_annotations:
                        processed_audio_data[audio_base_filename] = segments_with_annotations
                    else:
                        print(f"    No segments found or error for {audio_base_filename}")
                else:
                    print(f"  No JSON annotation found for {filename} in specified JSON folders. Skipping.")

print("\nProcessing complete.")
print(f"Total unique audio files processed with annotations: {len(processed_audio_data)}")

# Display a sample of the processed data to verify
if processed_audio_data:
    sample_key = next(iter(processed_audio_data))
    print(f"\nSample of processed data for '{sample_key}':")
    for i, segment_info in enumerate(processed_audio_data[sample_key]):
        print(f"  Segment {i+1}: Type={segment_info['event_type']}, Start={segment_info['start_ms']}ms, End={segment_info['end_ms']}ms, Segment_shape={segment_info['segment'].shape}")
    print("\nNote: 'segment' contains numpy arrays, which are not fully printed for brevity.")
else:
    print("No audio data was processed.")

# --- Code cell ---


# --- Code cell ---
def slice_audio_from_txt_annotations(audio_filepath, txt_filepath):
  """
  Slices an audio file into segments based on event annotations from a TXT file.

  Args:
    audio_filepath (str): Path to the audio (.wav) file.
    txt_filepath (str): Path to the corresponding TXT annotation file.

  Returns:
    list: A list of dictionaries, where each dictionary contains:
          - 'segment': The numpy array of the sliced audio segment.
          - 'event_type': The type of the event (e.g., 'Wheeze', 'Crackle', 'Normal').
          - 'start_ms': Start time of the event in milliseconds.
          - 'end_ms': End time of the event in milliseconds.
  """
  sliced_data = []

  try:
    y_full, sr_full = librosa.load(audio_filepath)
  except FileNotFoundError:
    print(f"Audio file not found: {audio_filepath}")
    return []
  except Exception as e:
    print(f"Error loading audio file {audio_filepath}: {e}")
    return []

  try:
    with open(txt_filepath, 'r') as f:
      # Skip header if present, or read all lines and filter if needed
      lines = f.readlines()
  except FileNotFoundError:
    print(f"TXT file not found: {txt_filepath}")
    return []

  for line in lines:
    parts = line.strip().split('\t') # Split by tab character
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

        start_ms = int(start_time_s * 1000)
        end_ms = int(end_time_s * 1000)

        start_sample = int(start_time_s * sr_full)
        end_sample = int(end_time_s * sr_full)

        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(y_full), end_sample)

        if start_sample < end_sample:
          y_segment = y_full[start_sample:end_sample]
          sliced_data.append({
              'segment': y_segment,
              'event_type': event_type,
              'start_ms': start_ms,
              'end_ms': end_ms
          })
      except ValueError:
        print(f"Skipping malformed line in {txt_filepath}: {line.strip()}")
  return sliced_data

# --- Code cell ---


# --- Code cell ---
# Path to the second dataset
second_dataset_path = '/home/ares/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/versions/2'
print(f"Processing second dataset: {second_dataset_path}")

# Iterate through the files in the directory
for root, _, files in os.walk(second_dataset_path):
    for filename in files:
        if filename.endswith('.wav'):
            audio_base_filename = os.path.splitext(filename)[0]
            audio_filepath = os.path.join(root, filename)
            txt_filepath = os.path.join(root, audio_base_filename + '.txt')

            if os.path.exists(txt_filepath):
                print(f"  Slicing {filename} with annotation from {os.path.basename(txt_filepath)}")
                segments_with_annotations = slice_audio_from_txt_annotations(audio_filepath, txt_filepath)
                if segments_with_annotations:
                    # Append to the existing processed_audio_data dictionary
                    # If a key already exists, this will overwrite it.
                    # If you want to merge, you might need a different structure.
                    processed_audio_data[audio_base_filename] = segments_with_annotations
                else:
                    print(f"    No segments found or error for {audio_base_filename}")
            else:
                print(f"  No TXT annotation found for {filename}. Skipping.")

print("\nSecond dataset processing complete.")
print(f"Total unique audio files processed including new dataset: {len(processed_audio_data)}")

# Display a sample of the processed data to verify
if processed_audio_data:
    sample_key = next(iter(processed_audio_data))
    print(f"\nSample of processed data for '{sample_key}':")
    for i, segment_info in enumerate(processed_audio_data[sample_key]):
        print(f"  Segment {i+1}: Type={segment_info['event_type']}, Start={segment_info['start_ms']}ms, End={segment_info['end_ms']}ms, Segment_shape={segment_info['segment'].shape}")
    print("\nNote: 'segment' contains numpy arrays, which are not fully printed for brevity.")
else:
    print("No audio data was processed.")

# --- Code cell ---


# --- Code cell ---
import librosa
import numpy as np
import random
import copy

def augment_audio_segment(segment, sr, max_augmentations=3):
    """
    Aplica random augmentation por elemento de audio individual, igualando las muestras inferiores.
    """
    augmented_segment = segment.copy() # Copia del segmento a aumentar
    num_augmentations = random.randint(1, max_augmentations) # Aplica un valor de aumento entre 1 y max_augmentations

    augmentation_types = [
        'add_noise',
        'pitch_shift'
    ] # Decía que eran las más usadas
    random.shuffle(augmentation_types)

    for i in range(num_augmentations):
        aug_type = augmentation_types[i % len(augmentation_types)] # Cycle through types

        if aug_type == 'add_noise':
            # Random Gaussian noise
            noise_amplitude = 0.005 * random.uniform(0.5, 2.0)
            augmented_segment += noise_amplitude * np.random.randn(len(augmented_segment))
        elif aug_type == 'pitch_shift':
            # Pitch shift with semitones
            n_steps = random.uniform(-1, 1) # -1 a 1 para que sea bajo
            augmented_segment = librosa.effects.pitch_shift(y=augmented_segment, sr=sr, n_steps=n_steps)

    return augmented_segment

# Convertir processed_audio_data en una lista de diccionarios por segmento
all_segments_list = []
for audio_file_name, segments in processed_audio_data.items():
    for segment_info in segments:
        all_segments_list.append(segment_info)

# COntar los segmentos originales
final_segments_list = list(all_segments_list)

current_event_counts_flat = Counter(seg['event_type'] for seg in final_segments_list)

print("Cantidad de eventos original:")
for event_type, count in current_event_counts_flat.most_common():
    print(f"- {event_type}: {count}")

# --- Code cell ---


# --- Code cell ---
target_count = 2000 # Definición del número final, se dejó en 4000 considerando la cantidad de muestras de la clase "Normal"


# Establecer clases que requieren augmentation (todas menos "Normal")
classes_to_augment = ['Wheeze', 'Fine Crackle', 'Coarse Crackle', 'Rhonchi', 'Wheeze+Crackle', 'Stridor', 'Crackle']
print(f"Clases a aumentar: {classes_to_augment}")

# Recopilar segmentos por tipo de evento para el muestreo a partir de la lista
segments_by_event_type_flat = {event_type: [] for event_type in current_event_counts_flat.keys()}
for segment_info in final_segments_list:
    segments_by_event_type_flat[segment_info['event_type']].append(segment_info)

augmented_segments_count = 0
sr_full = 22050 # pero depende de las anteriores celdas

for event_type in classes_to_augment:
    current_count = current_event_counts_flat[event_type]
    segments_for_type = segments_by_event_type_flat[event_type]

    needed_samples = target_count - current_count
    print(f"Aumentando {event_type}: Se requieren {needed_samples} samples.")

    for _ in range(needed_samples):
        original_segment_info = random.choice(segments_for_type)
        original_segment = original_segment_info['segment']

        # Aplicar augmentation
        augmented_wav = augment_audio_segment(original_segment, sr=sr_full)

        # Crear un nuevo segmento (no sé si sea necesario)
        # Mantener datos originales
        augmented_segment_info = {
            'segment': augmented_wav,
            'event_type': original_segment_info['event_type'],
            'start_ms': original_segment_info['start_ms'],
            'end_ms': original_segment_info['end_ms']
        }
        final_segments_list.append(augmented_segment_info)
        augmented_segments_count += 1

print(f"\nTotal augmented segments added: {augmented_segments_count}")

# Volver a calcular cantidad de eventos
post_augmentation_counts = Counter(seg['event_type'] for seg in final_segments_list)

print("\nEvent Type Counts After Augmentation:")
for event_type, count in post_augmentation_counts.most_common():
    print(f"- {event_type}: {count}")

# --- Code cell ---


# --- Code cell ---
import os
import json
import numpy as np # Import numpy

filename = f"segment_{i:06d}.json"
output_folder="/home/ares/Documents/BREATH/code/BreathIA/Augmented"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

filepath = os.path.join(output_folder, filename)

# Prepare data for JSON serialization
# Create a new list to avoid modifying the original final_segments_list if it's used elsewhere
serializable_segments = []
for segment_info in final_segments_list[0:4000]:
    # Make a copy of the dictionary to avoid modifying the original list elements
    temp_segment_info = segment_info.copy()
    if isinstance(temp_segment_info.get('segment'), np.ndarray):
        temp_segment_info['segment'] = temp_segment_info['segment'].tolist()
    serializable_segments.append(temp_segment_info)

with open(filepath, "w") as f:
  json.dump(serializable_segments, f)

