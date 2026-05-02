from WEEF2 import run_slicing
from Test import run_analysis
from WEEF5 import run_augmentation

OUTPUT_FOLDER = "/home/ares/Documents/BREATH/code/Segments_wav_4"
AUGMENTED_FOLDER = "/home/ares/Documents/BREATH/code/Augmented_Segments_wav_4"

def main():
    print("=" * 50)
    print("STARTING PIPELINE")
    print("=" * 50)

    DO_SLICING=True

    DO_ANALYSIS_SEGMENTS = True

    DO_AUGMENTATION=True

    DO_ANALYSIS_AUGMENTED_SEGMENTS= True


    if DO_SLICING:
        print("Slicing...")
        total_segments = run_slicing(OUTPUT_FOLDER)
        print(f"Generated segments: {total_segments}")

    if DO_ANALYSIS_SEGMENTS:
        print(f"Analyzing folder {OUTPUT_FOLDER}")
        run_analysis(OUTPUT_FOLDER)

    if DO_AUGMENTATION:
        print(f"Doing augmentation folder {OUTPUT_FOLDER}")
        run_augmentation(input_folder=OUTPUT_FOLDER, output_folder=AUGMENTED_FOLDER, target_count=1000)

    if DO_ANALYSIS_AUGMENTED_SEGMENTS:
        print(f"Analyzing folder {AUGMENTED_FOLDER}")
        run_analysis(AUGMENTED_FOLDER)
    

if __name__ == "__main__":
    main()