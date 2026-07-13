import os
import sys
import numpy as np
import tensorflow as tf
import config
from preprocess import wav_to_features

def load_labels():
    if os.path.exists(config.LABELS_PATH):
        with open(config.LABELS_PATH) as f:
            return [line.strip() for line in f if line.strip()]
    return config.BIRD_CLASSES

def predict_file(model, labels, wav_path):
    features = wav_to_features(wav_path)          # shape (N_MELS, FIXED_FRAMES)
    features = features[np.newaxis, ..., np.newaxis]  # -> (1, N_MELS, FIXED_FRAMES, 1)
    probs = model.predict(features, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return labels[top_idx], float(probs[top_idx]) * 100, probs

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py path/to/file.wav        (predict one file)")
        print("  python predict.py sample_audio/            (predict every .wav in a folder)")
        return

    if not os.path.exists(config.MODEL_PATH):
        print(f"[!] No trained model found at {config.MODEL_PATH}")
        print("    Run `python train.py` first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(config.MODEL_PATH)
    labels = load_labels()

    target = sys.argv[1]
    if os.path.isdir(target):
        wav_files = sorted(
            os.path.join(target, f) for f in os.listdir(target) if f.lower().endswith(".wav")
        )
    else:
        wav_files = [target]

    if not wav_files:
        print(f"No .wav files found in {target}")
        return

    for wav_path in wav_files:
        print(f"\nAudio: {wav_path}")
        try:
            bird, confidence, probs = predict_file(model, labels, wav_path)
            print(f"Prediction: {bird}")
            print(f"Confidence: {confidence:.1f}%")
        except Exception as e:
            print(f"[!] Could not process this file: {e}")


if __name__ == "__main__":
    main()
