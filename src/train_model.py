"""
train_model.py

Trains an LSTM classifier on the landmark sequences saved by collect_data.py
(or converted from a downloaded dataset like WLASL -- see README).

Expects a directory structure:
    data/sequences/<word>/0.npy
    data/sequences/<word>/1.npy
    ...
    data/sequences/<other_word>/0.npy
    ...

Each .npy file has shape (seq_length, 258).

Usage:
    python train_model.py --data_dir ../data/sequences --epochs 100
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def load_dataset(data_dir):
    words = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {word: idx for idx, word in enumerate(words)}

    X, y = [], []
    for word in words:
        word_dir = os.path.join(data_dir, word)
        for fname in os.listdir(word_dir):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(word_dir, fname))
                X.append(seq)
                y.append(label_map[word])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, label_map


def build_model(seq_length, feature_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(seq_length, feature_dim)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/sequences")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", default="../models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, label_map = load_dataset(args.data_dir)
    if len(X) == 0:
        raise SystemExit(f"No .npy sequences found under {args.data_dir}. Run collect_data.py first.")

    print(f"Loaded {len(X)} sequences across {len(label_map)} classes: {list(label_map.keys())}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = build_model(seq_length=X.shape[1], feature_dim=X.shape[2], num_classes=len(label_map))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    model_path = os.path.join(args.out_dir, "sign_lstm_model.keras")
    labels_path = os.path.join(args.out_dir, "label_map.json")
    model.save(model_path)
    # invert so index -> word for easy lookup at inference time
    with open(labels_path, "w") as f:
        json.dump({v: k for k, v in label_map.items()}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved label map to {labels_path}")


if __name__ == "__main__":
    main()