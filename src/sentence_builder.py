"""
sentence_builder.py

Core logic shared by both the CLI script and the web UI.

Expects a folder structure of:
    <root>/
        <word_folder_1>/
            01_x.jpg
            02_y.jpg
            ...
        <word_folder_2>/
            01_x.jpg
            ...

Each subfolder = one word. Letters within a subfolder are ordered by a
leading number in the filename and classified individually, then
concatenated (no separator) to form the word. Subfolders are themselves
ordered by a leading number in the folder name, and joined with spaces
to form the final sentence.

This keeps the "ordering" problem dead simple: it's just whatever order
you drag files into folders in, expressed as numeric prefixes -- no
special "space" gesture or delimiter image needed.

The overall result also includes `sentence_confidence`, the average
per-letter confidence across the whole sentence -- a single "how sure is
the model, overall" number, distinct from the individual per-letter
confidences. Letters predicted as J or Z are flagged with
`motion_warning: True`, since both are motion-based in real ASL
fingerspelling and a static image can only approximate them.
"""

import os
import re

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# J and Z are motion-based in real ASL fingerspelling (J traces a hook,
# Z traces the letter shape in the air) -- a single static image can only
# approximate a mid-motion pose, so predictions of these letters are
# structurally less trustworthy even at high confidence. We flag them
# rather than exclude them, since the common datasets (e.g. ASL Alphabet)
# do include static approximations for both.
MOTION_LETTERS = {"j", "z"}


def natural_sort_key(name):
    match = re.match(r"^(\d+)", name)
    return int(match.group(1)) if match else float("inf")


def classify_image(image_path, model, extractor):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, 0.0
    vector, _ = extractor.extract(frame)
    vector = np.expand_dims(vector, axis=0)
    probs = model.predict(vector, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return top_idx, float(probs[top_idx])


def build_sentence_from_folder(root_folder, model, label_map, extractor, confidence_threshold=0.5):
    """
    Returns a structured result:
    {
        "sentence": "hello world",
        "sentence_confidence": 0.91,
        "total_letters": 10,
        "low_confidence_count": 1,
        "motion_letters_used": ["z"],
        "words": [
            {
                "word_folder": "01_hello",
                "predicted_word": "hello",
                "letters": [
                    {"filename": "01_h.jpg", "letter": "h", "confidence": 0.94,
                     "low_confidence": False, "motion_warning": False},
                    ...
                ]
            },
            ...
        ]
    }
    Letters below confidence_threshold are still included but marked
    low_confidence=True instead of silently dropped, so the caller can
    flag them in the UI. Letters predicted as J or Z get
    motion_warning=True regardless of confidence.
    """
    word_folders = [
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ]
    word_folders.sort(key=natural_sort_key)

    result_words = []
    sentence_parts = []
    all_confidences = []
    motion_letters_used = []

    for word_folder in word_folders:
        folder_path = os.path.join(root_folder, word_folder)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        image_files.sort(key=natural_sort_key)

        letters = []
        word_chars = []
        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            top_idx, confidence = classify_image(img_path, model, extractor)
            if top_idx is None:
                continue
            letter = label_map[top_idx]
            low_conf = confidence < confidence_threshold
            is_motion_letter = letter.lower() in MOTION_LETTERS

            letters.append({
                "filename": fname,
                "letter": letter,
                "confidence": round(confidence, 4),
                "low_confidence": low_conf,
                "motion_warning": is_motion_letter,
            })
            word_chars.append(letter)
            all_confidences.append(confidence)
            if is_motion_letter:
                motion_letters_used.append(letter)

        predicted_word = "".join(word_chars)
        result_words.append({
            "word_folder": word_folder,
            "predicted_word": predicted_word,
            "letters": letters,
        })
        sentence_parts.append(predicted_word)

    sentence_confidence = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0
    low_confidence_count = sum(1 for c in all_confidences if c < confidence_threshold)

    return {
        "sentence": " ".join(sentence_parts),
        "sentence_confidence": sentence_confidence,
        "total_letters": len(all_confidences),
        "low_confidence_count": low_confidence_count,
        "motion_letters_used": sorted(set(motion_letters_used)),
        "words": result_words,
    }