import os
import tensorflow as tf
import config
TFLITE_PATH = os.path.join(config.BASE_DIR, "bird_model.tflite")
def main():
    if not os.path.exists(config.MODEL_PATH):
        print(f" No trained model found at {config.MODEL_PATH}")
        print("    Run train.py first.")
        return

    print(f"Loading {config.MODEL_PATH} ...")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    print("Converting to TensorFlow Lite (with quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Dynamic range quantization: shrinks weights to 8-bit where possible
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    keras_size = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

    print(f"\nDone.")
    print(f"  {config.MODEL_PATH}: {keras_size:.2f} MB")
    print(f"  {TFLITE_PATH}: {tflite_size:.2f} MB  ({keras_size/tflite_size:.1f}x smaller)")
    print("\nNext: run `python predict_tflite.py sample_audio/some_file.wav` to test it,")
    print("and check it agrees with predict.py's predictions before trusting it on the Pi.")

if __name__ == "__main__":
    main()
