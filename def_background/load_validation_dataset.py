import os
import json
import time
import numpy as np
import soundfile as sf

from scipy.signal import resample

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = (
    r"C:\Users\Oficina 01\Documents\Breath\SPRSound\BioCAS2025"
)

OUTPUT_FOLDER = (
    r"C:\Users\Oficina 01\Documents\Breath\BreathIA\validation_dataset_with_rules"
)

WINDOW_SIZE = 3.0

DEAD_TIME = 1.0

TARGET_SR = 22050

# =========================================================
# CLEAN OUTPUT FOLDER
# =========================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\nCleaning output folder...\n")

for file in os.listdir(OUTPUT_FOLDER):

    if file.endswith(".wav"):

        os.remove(
            os.path.join(
                OUTPUT_FOLDER,
                file
            )
        )

# =========================================================
# RULE FUNCTION
# =========================================================

def time_arrays(json_input,
                window_size=3.0,
                dead_time=1.0):

    # -----------------------------------------------------
    # LOAD JSON
    # -----------------------------------------------------

    if isinstance(json_input, str):

        with open(json_input, "r") as f:

            json_data = json.load(f)

    else:

        json_data = json_input

    # -----------------------------------------------------
    # EVENTS
    # -----------------------------------------------------

    events = []

    for e in json_data.get(
        "event_annotation",
        []
    ):

        start = float(e["start"]) / 1000.0

        end = float(e["end"]) / 1000.0

        duration = end - start

        event_type = e["type"]

        # -------------------------------------------------
        # ONLY VALID LABELS
        # -------------------------------------------------

        if event_type not in [
            "Normal",
            "Wheeze",
            "Fine Crackle"
        ]:
            continue

        # -------------------------------------------------
        # MINIMUM DURATION
        # -------------------------------------------------

        if event_type == "Wheeze":

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

    # -----------------------------------------------------
    # EMPTY
    # -----------------------------------------------------

    if len(events) == 0:
        return []

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    events.sort(
        key=lambda x: x["start"]
    )

    # -----------------------------------------------------
    # MIX DETECTION
    # -----------------------------------------------------

    for i in range(len(events) - 1):

        current = events[i]

        nxt = events[i + 1]

        gap = nxt["start"] - current["end"]

        # -------------------------------------------------
        # DIFFERENT CLASSES TOO CLOSE
        # -------------------------------------------------

        if (
            current["type"] != nxt["type"]
            and gap <= dead_time
        ):

            return []

    # -----------------------------------------------------
    # MERGE SAME CLASS
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
    # CREATE 3-SECOND WINDOWS
    # -----------------------------------------------------

    final = []

    half_window = window_size / 2

    for seg in merged:

        center = (
            seg["start"] + seg["end"]
        ) / 2

        window_start = center - half_window

        window_end = center + half_window

        # -------------------------------------------------
        # LEFT LIMIT
        # -------------------------------------------------

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
# INDEX WAV FILES
# =========================================================

print("\nIndexing WAV files...\n")

wav_index = {}

for root, dirs, files in os.walk(DATASET_ROOT):

    for file in files:

        if file.endswith(".wav"):

            wav_index[file] = os.path.join(
                root,
                file
            )

print(
    f"Indexed {len(wav_index)} wav files\n"
)

# =========================================================
# FIND JSON FILES
# =========================================================

json_files = []

for root, dirs, files in os.walk(DATASET_ROOT):

    for file in files:

        if file.endswith(".json"):

            json_files.append(
                os.path.join(root, file)
            )

print(
    f"Found {len(json_files)} json files\n"
)

# =========================================================
# PROCESS
# =========================================================

counter = 1

start_total = time.time()

for idx, json_path in enumerate(json_files):

    file_start = time.time()

    try:

        print("=" * 60)

        print(
            f"[{idx+1}/{len(json_files)}]"
        )

        print(f"JSON: {json_path}")

        # -------------------------------------------------
        # FIND WAV
        # -------------------------------------------------

        wav_name = os.path.basename(
            json_path
        ).replace(".json", ".wav")

        if wav_name not in wav_index:

            print("WAV NOT FOUND")

            continue

        wav_path = wav_index[wav_name]

        print(f"WAV: {wav_path}")

        # -------------------------------------------------
        # WINDOWS
        # -------------------------------------------------

        windows = time_arrays(json_path)

        print(
            f"Windows found: {len(windows)}"
        )

        if len(windows) == 0:

            print("DISCARDED")

            continue

        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------

        audio, sr = sf.read(wav_path)

        # Stereo -> mono
        if len(audio.shape) > 1:

            audio = audio.mean(axis=1)

        # -------------------------------------------------
        # RESAMPLE
        # -------------------------------------------------

        if sr != TARGET_SR:

            new_length = int(
                len(audio)
                * TARGET_SR
                / sr
            )

            audio = resample(
                audio,
                new_length
            )

            sr = TARGET_SR

        # -------------------------------------------------
        # SAVE SEGMENTS
        # -------------------------------------------------

        saved_count = 0

        for (
            annotation,
            start_sec,
            end_sec
        ) in windows:

            start_sample = int(
                start_sec * sr
            )

            end_sample = int(
                end_sec * sr
            )

            # ---------------------------------------------
            # OUT OF RANGE
            # ---------------------------------------------

            if end_sample > len(audio):

                print(
                    "Window exceeds audio"
                )

                continue

            segment = audio[
                start_sample:end_sample
            ]

            expected_length = int(
                WINDOW_SIZE * sr
            )

            # ---------------------------------------------
            # BAD LENGTH
            # ---------------------------------------------

            if len(segment) != expected_length:

                print(
                    f"Bad length: "
                    f"{len(segment)}"
                )

                continue

            # ---------------------------------------------
            # NORMALIZE
            # ---------------------------------------------

            segment = segment / (
                np.max(np.abs(segment))
                + 1e-9
            )

            clean_label = (
                annotation
                .replace(" ", "_")
            )

            filename = (
                f"sample_"
                f"{counter:06d}_"
                f"{clean_label}.wav"
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

        elapsed = (
            time.time() - file_start
        )

        print(
            f"Saved: {saved_count}"
        )

        print(
            f"Time: {elapsed:.2f} sec"
        )

    except Exception as e:

        print("\nERROR PROCESSING FILE")

        print(json_path)

        print(e)

# =========================================================
# DONE
# =========================================================

total_elapsed = (
    time.time() - start_total
)

print("\n" + "=" * 60)

print("DONE")

print(
    f"Total time: "
    f"{total_elapsed:.2f} sec"
)

print(
    f"Total saved: "
    f"{counter-1}"
)