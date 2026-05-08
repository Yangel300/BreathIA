import os
import random
import shutil
from collections import defaultdict



input_folder = r"C:\Users\ADMIN\Documents\BreathIA\concatenate"

output_folder = r"C:\Users\ADMIN\Documents\BreathIA\balanced_dataset"

target_normal_count = 4000

os.makedirs(output_folder, exist_ok=True)



files_by_label = defaultdict(list)

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(".wav"):
        continue

    # Example:
    # concat_000002_Crackle.wav

    name_without_ext = os.path.splitext(filename)[0]

    parts = name_without_ext.split("_")

    label = "_".join(parts[2:])

    filepath = os.path.join(input_folder, filename)

    files_by_label[label].append(filepath)



balanced_files = []

for label, file_list in files_by_label.items():

    print(f"{label}: {len(file_list)}")

    if label == "Normal":

        # Randomly select only target amount
        selected = random.sample(
            file_list,
            min(target_normal_count, len(file_list))
        )

        balanced_files.extend(selected)

        print(f" -> Undersampled to {len(selected)}")

    else:
        balanced_files.extend(file_list)



for filepath in balanced_files:

    filename = os.path.basename(filepath)

    destination = os.path.join(output_folder, filename)

    shutil.copy(filepath, destination)

print("\nBalanced dataset created.")
print(f"Saved in: {output_folder}")