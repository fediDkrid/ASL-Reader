"""
collect_static_images.py

Captures single-frame images to build or extend a static-pose dataset
used by train_static_model.py -- works for letters, numbers, or any
other held-pose sign. Saves plain image files under:
    data/static_images/<label>/<n>.jpg

Usage:
    python collect_static_images.py --label 5 --num_images 60

Repeat for every label you want to add (numbers, extra letters, etc.),
varying hand angle/distance/lighting a bit between shots so the model
generalizes instead of memorizing one exact pose.

Controls while running:
    SPACE  - capture the current frame (skipped automatically if no
             hand is detected, so you don't collect blank/zero images)
    q      - quit early

Images are saved from the RAW (unflipped) camera frame, matching how
live_letter_recognition.py extracts landmarks from the unflipped frame
-- so training data and live inference stay in the same orientation.
Only the on-screen preview is mirrored, for a natural "looking in a
mirror" feel while you pose.
"""

import argparse
import os

import cv2

from landmark_utils import LandmarkExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Class label for this batch, e.g. '5' or 'hello'")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--data_dir", default="../data/static_images")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor(static_image_mode=True)  # just for live "hand detected" feedback

    captured = 0
    print(f"Capturing images for label '{args.label}'. SPACE = capture, q = quit.")

    while captured < args.num_images:
        ok, frame = cap.read()
        if not ok:
            break

        vector, _ = extractor.extract(frame)
        hand_detected = bool(vector.any())

        preview = cv2.flip(frame.copy(), 1)
        status = "hand detected" if hand_detected else "no hand"
        color = (0, 255, 0) if hand_detected else (0, 165, 255)
        cv2.putText(preview, f"Label: {args.label} | Captured: {captured}/{args.num_images} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(preview, "SPACE = capture, q = quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Static Image Collection", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:  # SPACE
            if not hand_detected:
                print("  Skipped -- no hand detected in frame.")
                continue
            save_path = os.path.join(out_dir, f"{existing + captured}.jpg")
            cv2.imwrite(save_path, frame)  # save the RAW unflipped frame
            captured += 1
            print(f"Saved {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"Done. {captured} image(s) saved for '{args.label}'.")


if __name__ == "__main__":
    main()