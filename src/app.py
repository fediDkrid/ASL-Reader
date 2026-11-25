"""
app.py

Local web UI for the local-image sentence mode. Lets you pick an entire
folder from your browser's file picker (subfolders and all, via the
webkitdirectory attribute) and get back the assembled sentence.

Usage:
    cd src
    python app.py
Then open http://127.0.0.1:5000 in your browser.

How folder structure is preserved on upload:
    The browser's webkitdirectory input gives each selected file a
    `webkitRelativePath` like "my_sentence/01_hello/01_h.jpg". The
    frontend JS sends that path alongside each file. This script
    reconstructs the same folder tree in a temp directory server-side,
    then reuses the exact same sentence_builder logic as the CLI script.
"""

import json
import os
import shutil
import tempfile

import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from landmark_utils import LandmarkExtractor
from sentence_builder import build_sentence_from_folder

MODEL_PATH = "../models/static_sign_model.keras"
LABELS_PATH = "../models/static_label_map.json"

app = Flask(__name__)

# Load once at startup rather than per-request.
_model = None
_label_map = None
_extractor = None


def get_model_resources():
    global _model, _label_map, _extractor
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"No static model found at {MODEL_PATH}. Run train_static_model.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            _label_map = {int(k): v for k, v in json.load(f).items()}
        _extractor = LandmarkExtractor(static_image_mode=True)
    return _model, _label_map, _extractor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    relative_paths = request.form.getlist("paths")

    if not files or len(files) != len(relative_paths):
        return jsonify({"error": "No files received, or paths/files mismatch."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="sign_upload_")
    try:
        for file_obj, rel_path in zip(files, relative_paths):
            # rel_path looks like "my_sentence/01_hello/01_h.jpg"
            # Strip the top-level folder name so tmp_dir itself becomes the root.
            parts = rel_path.split("/")
            if len(parts) > 1:
                parts = parts[1:]
            dest_path = os.path.join(tmp_dir, *parts)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            file_obj.save(dest_path)

        try:
            model, label_map, extractor = get_model_resources()
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        confidence_threshold = float(request.form.get("confidence_threshold", 0.5))
        result = build_sentence_from_folder(
            tmp_dir, model, label_map, extractor, confidence_threshold
        )
        return jsonify(result)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)