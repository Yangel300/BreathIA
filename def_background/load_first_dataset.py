import os
import json
import time
import soundfile as sf

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = r"C:\Users\Oficina 01\Documents\Breath\SPRSound"

OUTPUT_FOLDER = r"C:\Users\Oficina 01\Documents\Breath\BreathIA\raw_dataset_with_rules"

WINDOW_SIZE = 3.0
DEAD_TIME = 1.0
TARGET_SR = 22050

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# RULE FUNCTION
# =========================================================

def time_arrays(json_input,
                window_size=3.0,
                dead_time=1.0):

    if isinstance(json_input, str):

        with open(json_input, "r") as f:
            json_data = json.load(f)

    else:
        json_data = json_input

    events = []

    for e in json_data.get("event_annotation", []):

        start = float(e["start"]) / 1000.0
        end = float(e["end"]) / 1000.0

        duration = end - start

        event_type = e["type"]

        # -------------------------------------------------
        # Minimum duration
        # -------------------------------------------------

        if event_type.lower() == "wheeze":
            min_duration = 0.7
        else:
            min_duration = 0.8

        if duration < min_duration:
            continue

        events.append({
            "type": event_type,
            "start": start,
            "end": end
        })

    if len(events) == 0:
        return []

    events.sort(key=lambda x: x["start"])

    # -----------------------------------------------------
    # MIX DETECTION
    # -----------------------------------------------------

    for i in range(len(events) - 1):

        current = events[i]
        nxt = events[i + 1]

        gap = nxt["start"] - current["end"]

        if (
            current["type"] != nxt["type"]
            and gap <= dead_time
        ):
            return []

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    merged = []

    current = events[0].copy()

    for nxt in events[1:]:

        gap = nxt["start"] - current["end"]

        if (
            nxt["type"] == current["type"]
            and gap <= dead_time
        ):

            current["end"] = nxt["end"]

        else:

            merged.append(current)
            current = nxt.copy()

    merged.append(current)

    # -----------------------------------------------------
    # CENTERED WINDOWS
    # -----------------------------------------------------

    final = []

    half_window = window_size / 2

    for seg in merged:

        center = (seg["start"] + seg["end"]) / 2

        window_start = center - half_window
        window_end = center + half_window

        if window_start < 0:
            window_start = 0
            window_end = window_size

        final.append([
            seg["type"],
            round(window_start, 3),
            round(window_end, 3)
        ])

    return final


# =========================================================
# INDEX ALL WAV FILES FIRST
# =========================================================

print("\nINDEXING WAV FILES...\n")

wav_index = {}

for root, dirs, files in os.walk(DATASET_ROOT):

    for file in files:

        if file.endswith(".wav"):

            wav_index[file] = os.path.join(root, file)

print(f"Indexed {len(wav_index)} wav files\n")

# =========================================================
# FIND JSON FILES
# =========================================================

json_files = []

for root, dirs, files in os.walk(DATASET_ROOT):

    for file in files:

        if file.endswith(".json"):

            json_files.append(os.path.join(root, file))

print(f"Found {len(json_files)} json files\n")

# =========================================================
# PROCESS
# =========================================================

counter = 1

start_total = time.time()

for idx, json_path in enumerate(json_files):

    file_start = time.time()

    try:

        print("=" * 60)
        print(f"[{idx+1}/{len(json_files)}]")
        print(f"JSON: {json_path}")

        # -------------------------------------------------
        # Find WAV
        # -------------------------------------------------

        wav_name = os.path.basename(json_path).replace(".json", ".wav")

        if wav_name not in wav_index:

            print("WAV NOT FOUND")
            continue

        wav_path = wav_index[wav_name]

        print(f"WAV: {wav_path}")

        # -------------------------------------------------
        # Generate windows
        # -------------------------------------------------

        windows = time_arrays(json_path)

        print(f"Windows found: {len(windows)}")

        if len(windows) == 0:

            print("DISCARDED")
            continue

        # -------------------------------------------------
        # Load audio
        # -------------------------------------------------

        print("Loading audio...")

        audio, sr = sf.read(wav_path)

        # Stereo -> mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != TARGET_SR:

            from scipy.signal import resample

            new_length = int(len(audio) * TARGET_SR / sr)

            audio = resample(audio, new_length)

            sr = TARGET_SR

        print(f"Audio loaded | Samples: {len(audio)}")

        # -------------------------------------------------
        # Save windows
        # -------------------------------------------------

        saved_count = 0

        for annotation, start_sec, end_sec in windows:

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            if end_sample > len(audio):

                print("Window exceeds audio length")
                continue

            segment = audio[start_sample:end_sample]

            expected_length = int(WINDOW_SIZE * sr)

            if len(segment) != expected_length:

                print(
                    f"Bad segment length: {len(segment)} "
                    f"(expected {expected_length})"
                )

                continue

            clean_label = annotation.replace(" ", "_")

            filename = (
                f"sample_{counter:06d}_{clean_label}.wav"
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                filename
            )

            sf.write(
                output_path,
                segment,
                sr
            )

            saved_count += 1
            counter += 1

        elapsed = time.time() - file_start

        print(f"Saved segments: {saved_count}")
        print(f"File time: {elapsed:.2f} sec")

    except Exception as e:

        print("\nERROR PROCESSING FILE")
        print(json_path)
        print(e)

total_elapsed = time.time() - start_total

print("\n" + "=" * 60)
print("DONE")
print(f"Total time: {total_elapsed:.2f} sec")
print(f"Total saved: {counter-1}")