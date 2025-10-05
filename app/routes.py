from flask import Blueprint, render_template, request, jsonify
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from . import model

main = Blueprint('main', __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")
        image = image.resize((28, 28))
        image = img_to_array(image)
        image = image.reshape(1, 28, 28, 1)
        image = image.astype("float32") / 255.0

        preds = model.predict(image)
        digit = np.argmax(preds)
        confidence = float(np.max(preds)) * 100

        return jsonify({"digit": int(digit), "confidence": round(confidence, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})
