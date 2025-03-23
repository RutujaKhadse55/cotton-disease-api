from flask import Flask, request, jsonify
import tensorflow as tf  # ✅ Use full TensorFlow
from PIL import Image
import numpy as np
import cv2
import logging
import time
import os
import gdown  # ✅ For downloading from Google Drive

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("CottonDiseaseModel")

# ✅ Class labels
class_names = ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]

# ✅ Google Drive model link
GOOGLE_DRIVE_FILE_ID = "1eXO9JIUfFWMi1IVMJxwD59laQ_XGHVYI"
MODEL_PATH = "/models/cotton_disease_model.h5"  # Model storage location

# ✅ Load model globally
model = None

def download_model():
    """Downloads the model from Google Drive if not available."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50000:
        logger.info("📥 Downloading model from Google Drive...")
        gdrive_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure directory exists
            gdown.download(gdrive_url, MODEL_PATH, fuzzy=True, quiet=False)
            logger.info("✅ Model downloaded successfully!")

            # Verify file size
            if os.path.getsize(MODEL_PATH) < 50000:
                logger.error("❌ Model file is too small, possibly corrupted.")
                raise Exception("Model file may be incomplete or corrupted.")

        except Exception as e:
            logger.error("❌ Model download failed: %s", str(e))
            raise Exception("Model download failed")

def load_model():
    """Loads the TensorFlow .h5 model."""
    global model
    if model is None:
        download_model()
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error("❌ Error loading model: %s", str(e))
            raise Exception("Model load failed")

def is_cotton_leaf(image):
    """Checks if the image is a cotton leaf using color filtering."""
    image_np = np.array(image)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels_ratio = np.sum(mask > 0) / (image_np.shape[0] * image_np.shape[1])
    logger.debug(f"✅ Green Pixel Ratio: {green_pixels_ratio:.2f}")

    return green_pixels_ratio >= 0.15

@app.route("/")
def home():
    return "✅ Cotton Disease Prediction API (H5 Model) is Running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image prediction using TensorFlow (.h5)."""
    try:
        start_time = time.time()
        load_model()

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image = Image.open(request.files["file"]).convert("RGB")

        if not is_cotton_leaf(image):
            logger.info("❌ Prediction: Not a Cotton Leaf")
            return jsonify({"class": "Unknown (Not a Cotton Leaf)", "confidence": 0})

        # Preprocess image
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run model prediction
        predictions = model.predict(img_array)
        confidence = round(100 * float(np.max(predictions[0])), 2)

        if confidence < 75:
            return jsonify({"class": "Unknown (Possibly Non-Cotton Leaf)", "confidence": confidence})

        predicted_class = class_names[np.argmax(predictions[0])]
        end_time = time.time()

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "time": round(end_time - start_time, 2)
        })

    except Exception as e:
        logger.error("❌ Prediction failed: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
