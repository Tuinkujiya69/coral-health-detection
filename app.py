from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename

# ======================
# APP SETUP
# ======================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================
# LOAD MODEL (ONLY ONCE)
# ======================
model = tf.keras.models.load_model("model/coral_model.h5")

# Class labels (modify based on your training)
classes = [" Bleached Coral", "Dead Coral", "Healthy Coral"]


# ======================
# HOME ROUTE
# ======================
@app.route("/")
def home():
    return render_template("index.html")


# ======================
# PREDICTION ROUTE
# ======================
@app.route("/predict", methods=["POST"])
def predict():

    # Check if file exists
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Check empty filename
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Secure filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Save file
    file.save(filepath)

    # ======================
    # IMAGE PREPROCESSING
    # ======================
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))  # ⚠️ must match training size

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ======================
    # MODEL PREDICTION
    # ======================
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "label": predicted_class,
        "confidence": round(confidence, 2)
    })


# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    app.run(debug=True)