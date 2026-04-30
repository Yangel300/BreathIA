import os
from collections import Counter

folder = r"C:\Users\ADMIN\Documents\BREATH-IA\segments_wav_3"

# Known classes
EVENT_TYPES = [
    "Normal",
    "Wheeze",
    "Crackle",
    "Wheeze+Crackle",
    "Rhonchi",
    "Stridor",
    "Fine Crackle",
    "Coarse Crackle"
]

counter = Counter()
unknown = 0

for file in os.listdir(folder):
    if file.endswith(".wav"):
        found = False
        
        for event in EVENT_TYPES:
            if event in file:
                counter[event] += 1
                found = True
                break
        
        if not found:
            unknown += 1

print("\nCounts per class:")
for k, v in counter.items():
    print(f"{k}: {v}")

print(f"\nUnknown/unmatched files: {unknown}")
print(f"Total files: {sum(counter.values()) + unknown}")