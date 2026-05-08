import os
import librosa
import numpy as np
import pickle


concatenate_folder = r"C:\Users\ADMIN\Documents\BreathIA\balanced_dataset"

sr = 22050
n_mels = 128
n_fft = 2048
hop_length = 512


def load_melspectrograms(folder_path):
    """
    Load WAV files, compute MelSpectrograms,
    and save them in a dictionary.

    Returns:
        dict:
        {
            "filename.wav": {
                "label": "Crackle",
                "melspectrogram": np.array
            }
        }
    """

    dataset = {}

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(folder_path, filename)

        try:
     

            y, _ = librosa.load(filepath, sr=sr)


            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )

            # Convert to dB
            mel_db = librosa.power_to_db(mel, ref=np.max)


            name_without_ext = os.path.splitext(filename)[0]

            parts = name_without_ext.split("_")

            label = "_".join(parts[2:])

            dataset[filename] = {
                "label": label,
                "melspectrogram": mel_db
            }

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return dataset



mel_dataset = load_melspectrograms(concatenate_folder)

print(f"\nTotal processed files: {len(mel_dataset)}")
