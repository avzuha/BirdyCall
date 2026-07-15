
import os
import sys
import numpy as np

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    # Lightweight runtime 
    from tflite_runtime.interpreter import Interpreter
import config
from preprocess import wav_to_features
TFLITE_PATH = os.path.join(config.BASE_DIR, "bird_model.tflite")

def load_labels():
    if os.path.exists(config.LABELS_PATH):
        with open(config.LABELS_PATH) as f:
            return [line.strip() for line in f if line.strip()]
    return config.BIRD_CLASSES

def predict_file(interpreter, input_details, output_details, labels, wav_path):
    features = wav_to_features(wav_path)
    features = features[np.newaxis, ..., np.newaxis].astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], features)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    top_idx = int(np.argmax(probs))
    return labels[top_idx], float(probs[top_idx]) * 100

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict_tflite.py path/to/file.wav")
        print("  python predict_tflite.py sample_audio/")
        return
    if not os.path.exists(TFLITE_PATH):
        print(f"[!] No .tflite model found at {TFLITE_PATH}")
        print("    Run `python convert_to_tflite.py` first.")
        return
    print("Loading TFLite model...")
    interpreter = Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
            bird, confidence = predict_file(interpreter, input_details, output_details, labels, wav_path)
            print(f"Prediction: {bird}")
            print(f"Confidence: {confidence:.1f}%")
        except Exception as e:
            print(f"Could not process this file: {e}")

if __name__ == "__main__":
    main()
