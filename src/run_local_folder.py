"""
run_local_folder.py

CLI fallback for local sentence recognition -- useful for testing without
spinning up the web UI. See app.py for the browser-based version.

Usage:
    python run_local_folder.py --folder ../data/static_images/my_sentence
"""

import argparse
import json
import os

import tensorflow as tf

from landmark_utils import LandmarkExtractor
from sentence_builder import build_sentence_from_folder

MODEL_PATH = "../models/static_sign_model.keras"
LABELS_PATH = "../models/static_label_map.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Root folder containing word subfolders")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No static model found at {MODEL_PATH}. Run train_static_model.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    extractor = LandmarkExtractor(static_image_mode=True)
    result = build_sentence_from_folder(
        args.folder, model, label_map, extractor, args.confidence_threshold
    )
    extractor.close()

    for word in result["words"]:
        letters_str = ", ".join(
            f"{l['letter']}({l['confidence']*100:.0f}%"
            f"{'  <-- low' if l['low_confidence'] else ''}"
            f"{'  [motion letter]' if l['motion_warning'] else ''})"
            for l in word["letters"]
        )
        print(f"{word['word_folder']:20s} -> '{word['predicted_word']}'   [{letters_str}]")

    print("\nAssembled sentence:")
    print(result["sentence"])
    print(f"\nSentence confidence (avg. across {result['total_letters']} letters): "
          f"{result['sentence_confidence']*100:.1f}%")
    if result["low_confidence_count"] > 0:
        print(f"  {result['low_confidence_count']} letter(s) below the confidence threshold.")
    if result["motion_letters_used"]:
        letters = ", ".join(l.upper() for l in result["motion_letters_used"])
        print(f"  Note: {letters} require motion in real ASL fingerspelling -- "
              f"static-image predictions for these are approximate.")


if __name__ == "__main__":
    main()