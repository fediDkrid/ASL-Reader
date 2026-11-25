

import json
import os

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# =============================================================================
# Landmark extraction (formerly landmark_utils.py)
# =============================================================================

mp_hands = mp.solutions.hands

NUM_HAND_LANDMARKS = 21
FEATURE_VECTOR_LENGTH = NUM_HAND_LANDMARKS * 3 * 2  # both hands, x/y/z each

# NOTE on detector choice: we use mp.solutions.hands directly rather than
# mp.solutions.holistic or a separate pose detector. Hands detects hands
# independently with no dependency on body pose, so it works on both
# cropped hand-only training images and full-body webcam frames. Body
# pose is deliberately NOT tracked: the ASL Alphabet training images are
# cropped to the hand only, so pose would always be an all-zero block
# during training, but live webcam frames show the full upper body --
# input the model would have never actually learned from, causing a
# train/test mismatch. Dropping pose removes that mismatch entirely.


class _Results:
    """Lightweight results container so draw_landmarks() keeps a stable
    interface regardless of how detection was performed internally."""

    __slots__ = ("left_hand_landmarks", "right_hand_landmarks")

    def __init__(self, left_hand_landmarks, right_hand_landmarks):
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks


class LandmarkExtractor:
    """Wraps a MediaPipe Hands instance. Reuse ONE instance across frames
    for performance -- do not create a new one per frame."""

    def __init__(self, static_image_mode: bool = False):
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, bgr_frame: np.ndarray):
        """Takes a single BGR frame (as read by cv2), returns a flat
        126-length float32 feature vector (hands only), plus a results
        object for optional skeleton drawing."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_results = self.hands.process(rgb)

        left_hand_lms = None
        right_hand_lms = None
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for lms, handedness in zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                if label == "Left" and left_hand_lms is None:
                    left_hand_lms = lms
                elif label == "Right" and right_hand_lms is None:
                    right_hand_lms = lms

        left_hand = self._flatten(left_hand_lms)
        right_hand = self._flatten(right_hand_lms)

        vector = np.concatenate([left_hand, right_hand]).astype(np.float32)
        results = _Results(left_hand_lms, right_hand_lms)
        return vector, results

    @staticmethod
    def _flatten(landmark_obj):
        if landmark_obj is None:
            return np.zeros(NUM_HAND_LANDMARKS * 3, dtype=np.float32)
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_obj.landmark], dtype=np.float32
        )
        # Normalize relative to the wrist so the feature vector describes
        # hand SHAPE, not where the hand happens to sit in the frame --
        # matters for signs that only differ by small relative
        # displacement (e.g. M vs N).
        wrist = pts[0].copy()
        pts -= wrist
        # Scale-normalize using wrist-to-middle-MCP distance so hand
        # size / distance-from-camera doesn't affect the features.
        scale = float(np.linalg.norm(pts[9]))
        if scale > 1e-6:
            pts /= scale
        return pts.flatten()

    def close(self):
        self.hands.close()


def draw_landmarks(bgr_frame, results):
    """Optional debug helper: draws the detected hand skeleton(s) on the frame."""
    mp_drawing = mp.solutions.drawing_utils
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(bgr_frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(bgr_frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return bgr_frame


# =============================================================================
# Motion-letter flag (the one thing pulled from sentence_builder.py)
# =============================================================================

# J and Z are motion-based in real ASL fingerspelling (J traces a hook,
# Z traces the letter shape in the air) -- a single still frame can only
# setting up motion letters, so we flag them here to avoid auto-appending them to the sentence.
MOTION_LETTERS = {"j", "z"}


# =============================================================================
# Live recognition loop (formerly live_letter_recognition.py)
# =============================================================================

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

        # Extract landmarks from the UNFLIPPED frame -- the ASL Alphabet
        # training images are not mirrored, so flipping before extraction
        # would mismatch x-coordinates (and can swap which hand MediaPipe
        # calls "Left"/"Right") relative to what the model learned.
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
            # nearest the origin in feature space.
            letter = None
            confidence = 0.0
            is_motion_letter = False
            stability_streak = 0
            last_letter = None

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