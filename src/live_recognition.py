"""
live_recognition.py

Runs the trained LSTM model against a live webcam feed continuously.
Maintains a rolling buffer of the last SEQ_LENGTH frames' landmarks and
re-predicts every STRIDE frames, overlaying the predicted word and its
confidence percentage on the video.

A simple stability filter avoids flickering: a prediction is only
"committed" (added to the running sentence) once it stays the top guess
for STABILITY_COUNT consecutive predictions above CONFIDENCE_THRESHOLD.

Usage:
    python live_recognition.py
"""

import collections
import json
import os

import cv2
import numpy as np
import tensorflow as tf

from landmark_utils import LandmarkExtractor, draw_landmarks

SEQ_LENGTH = 30          # must match seq_length used in collect_data.py / training
STRIDE = 5               # run a new prediction every N frames (lower = more responsive, more CPU)
CONFIDENCE_THRESHOLD = 0.70
STABILITY_COUNT = 3      # how many consecutive windows must agree before we "commit" a word

MODEL_PATH = "../models/sign_lstm_model.keras"
LABELS_PATH = "../models/label_map.json"


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"No model found at {MODEL_PATH}. Run collect_data.py to gather sequences, "
            "then train_model.py to train a model first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor(static_image_mode=False)

    frame_buffer = collections.deque(maxlen=SEQ_LENGTH)
    frame_count = 0

    current_word = ""
    current_confidence = 0.0
    last_committed = None
    stability_streak = 0
    sentence = []

    print("Live recognition running. Press 'c' to clear sentence, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        vector, results = extractor.extract(frame)
        frame_buffer.append(vector)
        frame_count += 1

        if len(frame_buffer) == SEQ_LENGTH and frame_count % STRIDE == 0:
            window = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
            probs = model.predict(window, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            current_word = label_map[top_idx]
            current_confidence = top_conf

            if top_conf >= CONFIDENCE_THRESHOLD:
                if current_word == last_committed:
                    stability_streak += 1
                else:
                    last_committed = current_word
                    stability_streak = 1

                if stability_streak == STABILITY_COUNT:
                    sentence.append(current_word)
                    stability_streak = 0  # reset so it needs to re-stabilize to repeat the same word
            else:
                stability_streak = 0
                last_committed = None

        display = draw_landmarks(frame.copy(), results)

        overlay_text = f"{current_word} ({current_confidence*100:.0f}%)" if current_word else "..."
        color = (0, 255, 0) if current_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(display, overlay_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        sentence_text = " ".join(sentence[-10:])  # show last 10 words
        cv2.putText(display, sentence_text, (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Sign Recognition", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence = []

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()