import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import config
from preprocess import build_dataset

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # CNN layers:
        # unchanged from original version, it extracts spacial features from the spectogram
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Long Short-Term Memory part, takes the cnn features and timesteos as inputs, LSTM1 analyzes the 1024 features across the 21 time steps, outputs 128 features per step, then LSTM2, analyzes 128 features across 21 time steps, outputs 64 feature vector
        # Reshape for LSTM: (16, 21, 64) -> (21 time_steps, 1024 features)
        layers.Reshape((21, 16*64)),

        # LSTM layers: learn temporal patterns in bird calls
        # LSTM1
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        # LSTM2
        layers.LSTM(64),
        layers.Dropout(0.2),

        # Dense layers: classification head
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Step 1/4: Building dataset from dataset/ folder...")
    X, y = build_dataset()
    print(f"  Total samples: {len(X)}  |  Classes present: {len(set(y.tolist()))}")
    X = X[..., np.newaxis]

    print("Step 2/4: Splitting into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Validation: {len(X_val)}")

    print("Step 3/4: Building and training the CNN...")
    model = build_model(
        input_shape=X_train.shape[1:], num_classes=len(config.BIRD_CLASSES)
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=16,
        callbacks=callbacks,
    )

    print("Step 4/4: Saving model and labels...")
    model.save(config.MODEL_PATH)
    with open(config.LABELS_PATH, "w") as f:
        f.write("\n".join(config.BIRD_CLASSES))

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nDone. Final validation accuracy: {val_acc*100:.1f}%")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Labels saved to: {config.LABELS_PATH}")
    print("\nNext: run `python predict.py sample_audio/some_bird.wav` to test it.")


if __name__ == "__main__":
    main()
