Project: Plant disease detection — AI assistant instructions

Purpose
- Short: Help a code-writing AI quickly understand, modify, and extend this Flask + Keras project.

Big picture
- This repo is a small Flask webapp that serves an image-based plant disease classifier.
- Model training and preprocessing live in `train_plant_disease.py` and `plant_disease_predict.py` (both build and save a Keras `.h5` model and write `class_names.json`).
- The web UI is in `templates/` and `static/`; runtime logic and HTTP endpoints are in `app.py` which loads `plant_disease_model.h5` and `class_names.json` from the repo root.

Key files to inspect first
- `app.py`: Flask routes (`/`, `/detect`) and the prediction flow. See `model_predict()` for image size, normalization, and how results are mapped to advice.
- `plant_disease_predict.py` and `train_plant_disease.py`: dataset checks, cleaning (`remove_bad_images`), ImageDataGenerator usage, model architecture, training loop, and saving (`plant_disease_model.h5` + `class_names.json`).
- `save_class_names.py`: quick script to generate `class_names.json` from a training folder if needed.
- `templates/` and `static/`: UI and upload handling; `static/uploads` is where uploaded images are stored at runtime.
- `plantvillage dataset/`: expected dataset layout: `train/` and `val/` subfolders with class-named directories.

Important conventions & patterns (project-specific)
- Image size and preprocessing: all code uses target size (128,128) and divides pixel values by 255.0.
- Class name format: directory names become labels; many use `___` as separators (displayed in UI via `class_label.replace("___", " - ")`).
- Model file and metadata: runtime expects `plant_disease_model.h5` and `class_names.json` at repo root.
- Disease mapping: `app.py` maps substrings (e.g., `scab`, `blight`, `bacterial`, `mosaic`) to categories (Fungal/Bacterial/Viral/Healthy) and looks up treatment/advice in `DISEASE_RECOMMENDATIONS`.
- Uploads: allowed extensions are `png,jpg,jpeg` and `app.config["MAX_CONTENT_LENGTH"] = 5MB`.
- Paths in training scripts are hard-coded Windows absolute paths (e.g., `C:\\Users\\...\\plantvillage dataset\\train`). Prefer refactoring to relative or env-configurable paths before running on other machines.

Developer workflows (exact commands and notes)
- Run the web app (development):
  - Ensure `plant_disease_model.h5` and `class_names.json` exist at repo root.
  - Start server: `python app.py` (runs Flask in debug mode).
- Train a model locally:
  - Place dataset under `plantvillage dataset/train` and `plantvillage dataset/val` (mirror current paths or edit scripts).
  - Run: `python train_plant_disease.py` (or `python plant_disease_predict.py`). These scripts will clean corrupt images, train a small CNN, and write `plant_disease_model.h5` and `class_names.json`.
  - Note: scripts use `epochs=5` for quick runs; increase for production.
- Generate `class_names.json` without training: `python save_class_names.py` (it reads the training folder and writes the JSON list).

Dependencies & environment
- Main libraries used: TensorFlow/Keras, Flask, Pillow (PIL), NumPy. There is no `requirements.txt` in the repo — install manually or create one.
- Suggested minimal install (example):
  - `python -m venv venv`
  - `venv\\Scripts\\activate` (Windows)
  - `pip install flask tensorflow pillow numpy`
- TF version: scripts assume Keras API from `tensorflow.keras`; use TensorFlow 2.x compatible runtime.

Debugging hints specific to this repo
- Model load errors: confirm `plant_disease_model.h5` matches the TensorFlow version used by the runtime; re-save the model with the runtime TF if needed.
- Missing classes or misaligned indices: ensure `class_names.json` was generated from the same training data and order as the model outputs (scripts write them using the same ImageDataGenerator ordering).
- Absolute paths in training scripts: update `train_dir` / `val_dir` to relative paths (or source from an env var) before running on other machines.
- Prediction mismatches: verify `model_predict()` uses the same image size and normalization as training (128x128, /255.0).

Change guidance for AI edits
- If editing training code, preserve `class_names.json` generation or ensure downstream `app.py` still reads the same format (a JSON array of class labels).
- If adding new model architectures, keep output layer size tied to `num_classes = len(train_generator.class_indices)` and save to `plant_disease_model.h5`.
- When changing upload behavior, update `ALLOWED_EXTENSIONS` and the `MAX_CONTENT_LENGTH` constant in `app.py`.

Where to look next (files)
- `app.py`
- `train_plant_disease.py`
- `plant_disease_predict.py`
- `save_class_names.py`

If anything above is unclear or you want this adapted (e.g., strict dependency pins, CI steps, or refactors to remove absolute paths), tell me which area to expand and I'll update this file.
