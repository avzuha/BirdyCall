import os
import sys
import numpy as np
import librosa
import config

def load_and_clean_audio(path):
    y, sr = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    # Trim silence from the start/end (top_db controls how strict this is;
    # lower = more aggressive trimming)
    y, _ = librosa.effects.trim(y, top_db=25)
    # Normalize volume so loud/quiet recordings are treated fairly
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Force fixed length: pad with silence if too short, cut if too long
    target_len = int(config.SAMPLE_RATE * config.CLIP_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y, sr

def audio_to_melspectrogram(y, sr):
    """Convert a 1D audio waveform into a 2D mel spectrogram (dB scale),
    resized to a fixed shape so every sample matches for the CNN."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Force fixed number of time frames
    frames = mel_db.shape[1]
    if frames < config.FIXED_FRAMES:
        pad_width = config.FIXED_FRAMES - frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :config.FIXED_FRAMES]

    # Scale to 0-1 range (helps the CNN train faster/more stably)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def wav_to_features(path):
    """One-stop function: WAV file path -> ready-to-use spectrogram array
    of shape (N_MELS, FIXED_FRAMES). This is the exact function predict.py
    uses too, so training and prediction always see data the same way."""
    y, sr = load_and_clean_audio(path)
    return audio_to_melspectrogram(y, sr)

def build_dataset(dataset_dir=config.DATASET_DIR, classes=config.BIRD_CLASSES):
    """Walk through dataset/<Bird_Name>/*.wav for every class, convert each
    file to a spectrogram, and return (X, y):
        X -> numpy array of shape (num_samples, N_MELS, FIXED_FRAMES)
        y -> numpy array of integer labels (index into `classes`)
    """
    X, y = [], []
    for label_idx, bird_name in enumerate(classes):
        folder = os.path.join(dataset_dir, bird_name)
        if not os.path.isdir(folder):
            print(f"  [!] Warning: folder not found, skipping: {folder}")
            continue

        wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
        print(f"  {bird_name}: {len(wav_files)} files")

        for fname in wav_files:
            fpath = os.path.join(folder, fname)
            try:
                features = wav_to_features(fpath)
                X.append(features)
                y.append(label_idx)
            except Exception as e:
                print(f"    [!] Failed on {fpath}: {e}")

    if not X:
        raise RuntimeError(
            "No audio files were found/processed."
        )
    X = np.array(X)
    y = np.array(y)
    return X, y
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        feats = wav_to_features(test_path)
        print(f"Loaded: {test_path}")
        print(f"Spectrogram shape: {feats.shape}  (should be ({config.N_MELS}, {config.FIXED_FRAMES}))")
        print(f"Value range: {feats.min():.3f} to {feats.max():.3f} (should be ~0 to 1)")
    else:
        print("Usage: python preprocess.py path/to/some_bird.wav")
        print("(This just tests preprocessing on a single file.)")
