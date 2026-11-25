"""
landmark_utils.py

Shared MediaPipe wrapper used by every other script in this project.
Both the live-camera pipeline and the static-image pipeline call this same
function, so the feature representation is guaranteed to be identical in
both modes.

We track:
  - 21 landmarks per hand (x, y, z) for both hands -> 21*3*2 = 126 values

Total feature vector length: 126

If a hand is not detected in a frame, that block is filled with zeros so
the vector length is always constant (required for feeding into a
neural net).

NOTE on detector choice:
    We use `mp.solutions.hands.Hands` directly rather than
    `mp.solutions.holistic.Holistic` or a separate `mp.solutions.pose.Pose`
    detector. `Hands` detects hands directly with no dependency on body
    pose, so it works on both cropped hand images and full-body frames.

    We deliberately do NOT track body pose. Pose landmarks in MediaPipe
    include head/face points (nose, eyes, ears) alongside the
    shoulders/arms, which:
      (1) tracks facial position for no benefit -- fingerspelling and
          sign recognition here only need hand shape, and
      (2) caused a real train/test mismatch: the ASL Alphabet training
          images are cropped to the hand only, so pose was always a
          constant all-zero block during training, but live webcam
          frames show the full upper body/face, so pose came back
          non-zero at inference -- input the model had never actually
          learned from. Dropping pose removes both the privacy concern
          and that mismatch, and should improve accuracy/precision.
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

NUM_HAND_LANDMARKS = 21
FEATURE_VECTOR_LENGTH = NUM_HAND_LANDMARKS * 3 * 2


class _Results:
    """Lightweight results container so draw_landmarks() and any other
    downstream code that expects `.left_hand_landmarks` /
    `.right_hand_landmarks` keeps working unchanged."""

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

    def extract(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, "_Results"]:
        """
        Takes a single BGR frame (as read by cv2), returns a flat
        126-length float32 feature vector (hands only -- no pose/face).
        """
        import cv2

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

        left_hand = self._flatten(left_hand_lms, NUM_HAND_LANDMARKS, 3)
        right_hand = self._flatten(right_hand_lms, NUM_HAND_LANDMARKS, 3)

        vector = np.concatenate([left_hand, right_hand]).astype(np.float32)
        results = _Results(left_hand_lms, right_hand_lms)
        return vector, results

    @staticmethod
    def _flatten(landmark_obj, count, dims):
        if landmark_obj is None:
            return np.zeros(count * dims, dtype=np.float32)
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_obj.landmark], dtype=np.float32
        )
        # Normalize relative to the wrist (landmark 0) so the feature
        # vector describes hand SHAPE, not where the hand happens to sit
        # in the frame. This matters most for signs that only differ by
        # a small relative displacement -- e.g. M vs N, which differ only
        # in whether the thumb tucks under two or three fingers. Without
        # this, that difference is tiny relative to frame-position noise.
        wrist = pts[0].copy()
        pts -= wrist
        # Scale-normalize using wrist-to-middle-MCP distance (landmark 9)
        # so hand size / distance-from-camera doesn't affect the features.
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