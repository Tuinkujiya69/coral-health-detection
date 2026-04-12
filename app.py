from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import cv2
import base64
import io

# App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load Model & Classes
model = tf.keras.models.load_model("model/coral_model.h5")

# Load class labels dynamically from training
try:
    with open("model/class_labels.json", "r") as f:
        class_indices = json.load(f)
    # Invert dict: {index: class_name}
    classes = {v: k for k, v in class_indices.items()}
    classes = [classes[i] for i in range(len(classes))]
except FileNotFoundError:
    # Fallback to default
    classes = ["Bleached Coral", "Dead Coral", "Healthy Coral", "Partially Bleached"]

print(f"Loaded model with classes: {classes}")


# GRAD-CAM Function
def generate_gradcam(img_array, model):
    """Generate Grad-CAM heatmap showing which regions influenced the prediction."""
    try:
        # Find the last conv layer in MobileNetV2
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
            # For Sequential with MobileNetV2 inside
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        last_conv_layer = sublayer.name
                        break
                if last_conv_layer:
                    break

        if last_conv_layer is None:
            return None

        # Build grad model
        # For Sequential with base model, access the MobileNetV2 submodel
        base_model = model.layers[0]
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv_layer).output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to image size and colorize
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Superimpose on original image
        original = np.uint8(img_array[0] * 255)
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Convert to base64 for JSON response
        _, buffer = cv2.imencode('.jpg', superimposed)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        return heatmap_b64

    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return None


# Home Route
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Image Preprocessing
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))   # Updated to 224x224 for MobileNetV2
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Model Prediction
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_class = classes[predicted_index]
    confidence = float(np.max(prediction))

    # All class probabilities
    all_probs = {classes[i]: round(float(prediction[0][i]) * 100, 1) for i in range(len(classes))}

    # Reject non-coral images
    CONFIDENCE_THRESHOLD = 0.75
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "error": "not_coral",
            "message": "This doesn't look like a coral image. Please upload a clear underwater coral photo."
        })

    # GRAD-CAM Heatmap
    heatmap_b64 = generate_gradcam(img_array, model)

    # Health Status Messages
    status_messages = {
        "Healthy Coral": "✅ Coral is in good health. No signs of bleaching detected.",
        "Bleached Coral": "⚠️ Coral bleaching detected. The coral has expelled its algae due to stress.",
        "Dead Coral": "❌ Coral appears to be dead. Immediate environmental attention needed.",
        "Partially Bleached": "🔶 Partial bleaching detected. Coral is under stress but may recover."
    }
    status_message = status_messages.get(predicted_class, "Classification complete.")

    return jsonify({
        "label": predicted_class,
        "confidence": round(confidence * 100, 1),
        "all_probabilities": all_probs,
        "status_message": status_message,
        "heatmap": heatmap_b64  # Base64 encoded Grad-CAM image
    })


if __name__ == "__main__":
    app.run(debug=True)