import os
import json
import numpy as np

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask_sqlalchemy import SQLAlchemy

# ================= APP SETUP ================= #
app = Flask(__name__)
app.secret_key = "plantcare_secret"

# ================= BASE DIR ================= #
basedir = os.path.abspath(os.path.dirname(__file__))

# ================= CONFIG ================= #
UPLOAD_FOLDER = os.path.join(basedir, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "messages.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ================= LOAD MODEL ================= #
MODEL_PATH = os.path.join(basedir, "plant_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(basedir, "class_names.json")

try:
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    print("✅ Model & class names loaded")
except Exception as e:
    print("❌ Model loading error:", e)
    model = None
    class_names = []

# ================= DISEASE INFO ================= #
disease_info = {
    "Potato___Early_blight": {
        "chemical": "Mancozeb 75% WP or Chlorothalonil",
        "advice": "Remove infected leaves. Avoid overhead irrigation. Spray Mancozeb every 7–10 days."
    },
    "Potato___Late_blight": {
        "chemical": "Metalaxyl + Mancozeb",
        "advice": "Apply fungicide immediately. Ensure good drainage and destroy infected plants."
    },
    "Tomato___Early_blight": {
        "chemical": "Chlorothalonil or Copper Fungicide",
        "advice": "Remove affected leaves. Rotate crops and avoid wet foliage."
    },
    "Tomato___Late_blight": {
        "chemical": "Metalaxyl-based fungicide",
        "advice": "Spray fungicide early. Avoid high humidity and overcrowding."
    },
    "Healthy": {
        "chemical": "No chemical required",
        "advice": "Plant is healthy. Maintain proper watering and nutrition."
    }
}

# ================= DATABASE MODEL ================= #
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    message = db.Column(db.Text)

# ================= HELPERS ================= #
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    label = class_names[idx]
    confidence = preds[idx] * 100
    status = "Healthy" if "healthy" in label.lower() else "Diseased"

    return label, confidence, status

# ================= ROUTES ================= #
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("detect.html", error="No image selected")

        if not allowed_file(file.filename):
            return render_template("detect.html", error="Invalid file type")

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        label, confidence, status = model_predict(path)

        # ✅ CORRECT LOOKUP
        info = disease_info.get(
            label,
            {
                "chemical": "Consult agriculture expert",
                "advice": "Follow standard crop protection practices."
            }
        )

        return render_template(
            "result.html",
            image_path=url_for("static", filename=f"uploads/{filename}"),
            class_label=label.replace("___", " - "),
            confidence=round(confidence, 2),
            status=status,
            chemical=info["chemical"],
            advice=info["advice"]
        )

    return render_template("detect.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    success = False
    if request.method == "POST":
        msg = ContactMessage(
            name=request.form.get("name"),
            email=request.form.get("email"),
            phone=request.form.get("phone"),
            message=request.form.get("message"),
        )
        db.session.add(msg)
        db.session.commit()
        success = True

    return render_template("contact.html", success=success)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
