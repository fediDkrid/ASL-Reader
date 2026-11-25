
"""
live_letter_recognition.py

Live webcam recognition for fingerspelled LETTERS (static hand poses),
using the same static classifier trained by train_static_model.py on the
ASL Alphabet dataset -- no separate motion-based model or recorded clips
needed for this mode.

Unlike live_recognition.py (which uses an LSTM over a sliding window of
frames to recognize whole-word motions), this script classifies each
frame independently, since a fingerspelled letter is a held pose, not a
movement.

Usage:
    python live_letter_recognition.py

Controls:
    SPACE      - insert a space (marks a word boundary in the sentence)
    BACKSPACE  - delete the last character
    c          - clear the whole sentence
    q          - quit

Letters are also auto-appended once the same prediction stays confident
and stable for a few consecutive frames, so you don't have to press a
key for every single letter -- just hold the pose steady for a moment.
"""

import collections
import json
import os

import cv2
import numpy as np
import tensorflow as tf

from landmark_utils import LandmarkExtractor, draw_landmarks
from sentence_builder import MOTION_LETTERS

MODEL_PATH = "../models/static_sign_model.keras"
LABELS_PATH = "../models/static_label_map.json"

CONFIDENCE_THRESHOLD = 0.75
STABILITY_COUNT = 12       # consecutive stable frames required before auto-appending
COOLDOWN_FRAMES = 15       # frames to wait after appending before the same letter can auto-append again


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"No static model found at {MODEL_PATH}. Run train_static_model.py first "
            "(after downloading the ASL Alphabet dataset into data/static_images/)."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor(static_image_mode=False)

    sentence = []
    last_letter = None
    stability_streak = 0
    cooldown = 0

    print("Live letter recognition running.")
    print("SPACE = add space | BACKSPACE = delete last char | c = clear | q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Extract landmarks from the UNFLIPPED frame. The ASL Alphabet
        # training images are not mirrored, so flipping before extraction
        # mismatches x-coordinates (and can swap which hand MediaPipe
        # calls "Left"/"Right") relative to what the model learned --
        # a common source of systematic letter confusion.
        vector, results = extractor.extract(frame)
        hand_detected = bool(np.any(vector))

        if cooldown > 0:
            cooldown -= 1

        if hand_detected:
            vector_batch = np.expand_dims(vector, axis=0)
            probs = model.predict(vector_batch, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            letter = label_map[top_idx]
            is_motion_letter = letter.lower() in MOTION_LETTERS

            # Stability tracking for auto-append
            if confidence >= CONFIDENCE_THRESHOLD:
                if letter == last_letter:
                    stability_streak += 1
                else:
                    last_letter = letter
                    stability_streak = 1

                if stability_streak == STABILITY_COUNT and cooldown == 0:
                    sentence.append(letter)
                    cooldown = COOLDOWN_FRAMES
                    stability_streak = 0
            else:
                stability_streak = 0
                last_letter = None
        else:
            # No hand detected -- don't force a classification. The model
            # never saw all-zero vectors during training (they're dropped
            # in train_static_model.py), so predicting on one is
            # meaningless and tends to collapse onto whatever class sits
            # nearest the origin in feature space -- a frequent source of
            # a "phantom" letter showing up constantly.
            letter = None
            confidence = 0.0
            is_motion_letter = False
            stability_streak = 0
            last_letter = None

        # --- Draw overlay ---
        # Draw the skeleton on the unflipped frame first (landmark
        # coordinates match this orientation), THEN flip the whole
        # annotated image for a natural mirror view, THEN add text --
        # adding text before the flip would render it backwards.
        skeleton = draw_landmarks(frame.copy(), results)
        display = cv2.flip(skeleton, 1)

        if letter is not None:
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            if is_motion_letter:
                color = (200, 80, 200)  # flag motion-based letters (J/Z) distinctly

            overlay_text = f"{letter} ({confidence*100:.0f}%)"
            if is_motion_letter:
                overlay_text += "  [motion letter - approx.]"
        else:
            color = (128, 128, 128)
            overlay_text = "No hand detected"
        cv2.putText(display, overlay_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Small progress bar toward auto-append, purely visual feedback
        bar_width = int(200 * (stability_streak / STABILITY_COUNT))
        cv2.rectangle(display, (10, 55), (10 + bar_width, 65), color, -1)
        cv2.rectangle(display, (10, 55), (210, 65), (255, 255, 255), 1)

        sentence_text = "".join(sentence)[-40:]  # show last 40 chars
        cv2.putText(display, sentence_text, (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Live Letter Recognition", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = []
        elif key == 32:  # SPACE
            sentence.append(" ")
        elif key == 8:  # BACKSPACE
            if sentence:
                sentence.pop()

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print("\nFinal sentence:")
    print("".join(sentence))


if __name__ == "__main__":
    main()
