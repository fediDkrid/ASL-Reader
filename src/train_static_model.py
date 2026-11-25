"""
train_static_model.py

Trains a classifier for STATIC single-frame signs (e.g. fingerspelled
letters, or any word whose sign is a held pose rather than a movement).

Expects a directory of labeled images:
    data/static_images/<label>/img1.jpg
    data/static_images/<label>/img2.jpg
    ...
    data/static_images/<other_label>/img1.jpg
    ...

This is the natural structure you get if you download the "ASL Alphabet"
or "Sign Language MNIST" Kaggle datasets and unzip them into
data/static_images/.

Usage:
    python train_static_model.py --data_dir ../data/static_images --epochs 50
"""

import argparse
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from landmark_utils import LandmarkExtractor, FEATURE_VECTOR_LENGTH


def load_dataset(data_dir, extractor):
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {label: idx for idx, label in enumerate(labels)}

    X, y = [], []
    total_zero_vectors = 0
    total_files = 0
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        files = [f for f in os.listdir(label_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total_files += len(files)
        zero_count = 0
        for fname in files:
            img = cv2.imread(os.path.join(label_dir, fname))
            if img is None:
                continue
            vector, _ = extractor.extract(img)
            if not np.any(vector):
                # No hand detected at all -- this is a blank/zero feature
                # vector with no signal in it. Including it would just
                # teach the network to associate "no hand visible" with
                # this label, which actively hurts accuracy. Drop it
                # instead of keeping it as a training example.
                zero_count += 1
                continue
            X.append(vector)
            y.append(label_map[label])
        total_zero_vectors += zero_count
        flag = "  <-- WARNING: many failed detections" if files and zero_count / len(files) > 0.15 else ""
        print(f"  {label}: {len(files)} images, {zero_count} with no hand detected (dropped){flag}")

    print(f"\nTotal: {total_files} images, {total_zero_vectors} dropped for no detection "
          f"({total_zero_vectors / total_files * 100:.1f}% of dataset), "
          f"{len(X)} usable training examples" if total_files else "")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), label_map


def build_model(feature_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(feature_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
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
    parser.add_argument("--data_dir", default="../data/static_images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="../models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Extracting landmarks from images (this can take a while for large datasets)...")
    extractor = LandmarkExtractor(static_image_mode=True)
    X, y, label_map = load_dataset(args.data_dir, extractor)
    extractor.close()

    if len(X) == 0:
        raise SystemExit(f"No images found under {args.data_dir}")

    print(f"\nLoaded {len(X)} images across {len(label_map)} classes")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(FEATURE_VECTOR_LENGTH, len(label_map))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    model_path = os.path.join(args.out_dir, "static_sign_model.keras")
    labels_path = os.path.join(args.out_dir, "static_label_map.json")
    model.save(model_path)
    with open(labels_path, "w") as f:
        json.dump({v: k for k, v in label_map.items()}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved label map to {labels_path}")


if __name__ == "__main__":
    main()