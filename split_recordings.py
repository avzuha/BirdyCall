import os
import sys
import numpy as np
import librosa
import soundfile as sf
import config

MIN_CLIP_SECONDS = 1.5
MAX_CLIP_SECONDS = 5.0
TOP_DB = 30

def split_file(input_path, output_folder, top_db=TOP_DB):
    os.makedirs(output_folder, exist_ok=True)
    y, sr = librosa.load(input_path, sr=config.SAMPLE_RATE, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    saved = 0
    for i, (start, end) in enumerate(intervals):
        duration = (end - start) / sr
        if duration < MIN_CLIP_SECONDS:
            continue
        max_samples = int(MAX_CLIP_SECONDS * sr)
        segment = y[start:min(end, start + max_samples)]
        out_path = os.path.join(output_folder, f"{base_name}_clip{i:03d}.wav")
        sf.write(out_path, segment, sr)
        saved += 1
    print(f"{input_path}: found {len(intervals)} segments, saved {saved} usable clips -> {output_folder}")
    return saved
def main():
    if len(sys.argv) < 3:
        print("Usage: python split_recordings.py path/to/raw_file.wav path/to/output_folder")
        print("       python split_recordings.py path/to/raw_folder/ path/to/output_folder")
        return
    input_path, output_folder = sys.argv[1], sys.argv[2]
    if os.path.isdir(input_path):
        wav_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(".wav")
        ]
        total = 0
        for f in wav_files:
            total += split_file(f, output_folder)
        print(f"\nDone. {total} total clips saved across {len(wav_files)} input files.")
    else:
        split_file(input_path, output_folder)
    print("\nNext: skim the output folder and delete/rename any bad clips, then run train.py.")
if __name__ == "__main__":
    main()
