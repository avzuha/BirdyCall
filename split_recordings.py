import os
import sys
import librosa
import soundfile as sf
import config

CLIP_SECONDS = 5.0  # fixed length

def split_file_fixed(input_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    y, sr = librosa.load(input_path, sr=config.SAMPLE_RATE, mono=True)
    total_samples = len(y)
    clip_samples = int(CLIP_SECONDS * sr)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    saved = 0

    # iterate in steps of 5 seconds
    for i in range(0, total_samples, clip_samples):
        segment = y[i:i+clip_samples]
        if len(segment) < clip_samples:  # skip last fragment if shorter
            break
        out_path = os.path.join(output_folder, f"{base_name}_clip{i//clip_samples:03d}.wav")
        sf.write(out_path, segment, sr)
        saved += 1

    print(f"{input_path}: saved {saved} clips of {CLIP_SECONDS}s each -> {output_folder}")
    return saved

def main():
    if len(sys.argv) < 3:
        print("Usage: python split_recordings.py path/to/raw_file.wav path/to/output_folder")
        return
    input_path, output_folder = sys.argv[1], sys.argv[2]
    if os.path.isdir(input_path):
        wav_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".wav")]
        total = 0
        for f in wav_files:
            total += split_file_fixed(f, output_folder)
        print(f"\nDone. {total} total clips saved across {len(wav_files)} input files.")
    else:
        split_file_fixed(input_path, output_folder)

if __name__ == "__main__":
    main()

