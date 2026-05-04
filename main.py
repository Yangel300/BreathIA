import os

# =========================================
# PATHS BASE
# =========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR  # main está en la raíz

# IMPORTS
from background.WEEF2 import run_slicing
from background.Test import run_analysis
from background.WEEF5 import run_augmentation

# =========================================
# RUTAS DE DATOS
# =========================================
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "Segments_wav_4")
AUGMENTED_FOLDER = os.path.join(PROJECT_ROOT, "Augmented_Segments_wav_4")

SPR_PATH = os.path.join(PROJECT_ROOT, "SPRSound")


# =========================================
# UTIL
# =========================================
def ensure_dir(path):
    if not os.path.exists(path):
        print(f"[INFO] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)


# =========================================
# MAIN
# =========================================
def main():
    print("=" * 50)
    print("STARTING PIPELINE")
    print("=" * 50)

    DO_SLICING = True
    DO_ANALYSIS_SEGMENTS = True
    DO_AUGMENTATION = True
    DO_ANALYSIS_AUGMENTED_SEGMENTS = True

    # Crear carpetas según los flags activos
    if DO_SLICING or DO_ANALYSIS_SEGMENTS:
        ensure_dir(OUTPUT_FOLDER)

    if DO_AUGMENTATION or DO_ANALYSIS_AUGMENTED_SEGMENTS:
        ensure_dir(AUGMENTED_FOLDER)

    try:
        if DO_SLICING:
            print("[STEP] Slicing...")
            total_segments = run_slicing(
                OUTPUT_FOLDER,
                sprsound_path=SPR_PATH
            )
            print(f"[OK] Generated segments: {total_segments}")

        if DO_ANALYSIS_SEGMENTS:
            print(f"[STEP] Analyzing folder: {OUTPUT_FOLDER}")
            run_analysis(OUTPUT_FOLDER)

        if DO_AUGMENTATION:
            print(f"[STEP] Augmenting from: {OUTPUT_FOLDER}")
            run_augmentation(
                input_folder=OUTPUT_FOLDER,
                output_folder=AUGMENTED_FOLDER,
                target_count=4000
            )

        if DO_ANALYSIS_AUGMENTED_SEGMENTS:
            print(f"[STEP] Analyzing folder: {AUGMENTED_FOLDER}")
            run_analysis(AUGMENTED_FOLDER)

    except Exception as e:
        print("[ERROR] Pipeline failed:")
        print(e)


if __name__ == "__main__":
    main()