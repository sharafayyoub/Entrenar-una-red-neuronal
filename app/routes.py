from flask import Blueprint, render_template, request, jsonify, current_app
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from .firebase_config import db


main = Blueprint('main', __name__)


# ==============================
# Página principal
# ==============================
@main.route("/")
def index():
    return render_template("index.html")


# ==============================
# Ruta de predicción
# ==============================
@main.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔔 /predict request received")
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No se ha enviado la imagen."})

        # Quitar encabezado base64
        image_data = data['image']
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decodificar imagen
        img = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")
        print(f"🖼 Imagen original size: {img.size}, mode: {img.mode}")

        # Determinar modo de redimensionado según versión de Pillow
        if hasattr(Image, "Resampling"):
            resample_mode = Image.Resampling.LANCZOS
        else:
            resample_mode = Image.ANTIALIAS

        # Redimensionar a 28x28
        img = img.resize((28, 28), resample_mode)

        # Convertir a array (grayscale)
        arr = img_to_array(img)
        print("📐 after img_to_array shape:", arr.shape, "min/max:", arr.min(), arr.max())

        # Normalizar a 0-1
        arr = arr.astype("float32") / 255.0

        # Asegurar batch y canales
        x1 = np.expand_dims(arr, axis=0)  # (1,28,28,1)
        x2 = np.transpose(x1, (0, 3, 1, 2))  # (1,1,28,28)

        # Obtener modelo
        model = current_app.config.get("MODEL", None)
        if model is None:
            print("❌ Modelo no cargado en current_app.config['MODEL']")
            return jsonify({"error": "Modelo no cargado en servidor."})

        # Intentar predecir
        preds = None
        try:
            print("▶ Probando predicción con shape", x1.shape)
            preds = model.predict(x1)
            print("✅ Predicción OK con channels_last")
        except Exception as e:
            print("⚠️ Fallo con channels_last:", e)
            try:
                print("▶ Probando predicción con shape", x2.shape)
                preds = model.predict(x2)
                print("✅ Predicción OK con channels_first")
            except Exception as e2:
                print("❌ Fallo predicción con ambos formatos:", e2)
                return jsonify({"error": f"Error en predict: {e2}"})

        preds = np.asarray(preds)
        if preds.ndim == 2:
            preds = preds[0]
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        print(f"🔎 Resultado: {digit} con confianza {confidence:.2f}%")

        # 🔥 Guardar en Firebase
        db.collection("predicciones").add({
            "resultado": str(digit),
            "confianza": confidence,
            "timestamp": np.datetime_as_string(np.datetime64('now'))
        })

        return jsonify({"digit": digit, "confidence": round(confidence, 2)})

    except Exception as e:
        print("❌ Exception en /predict:", e)
        return jsonify({"error": str(e)})


# ==============================
# Ruta para guardar datos (si deseas llamarla manualmente desde JS)
# ==============================
@main.route('/save', methods=['POST'])
def save_data():
    data = request.json
    if not data:
        return jsonify({"error": "No se enviaron datos"}), 400

    try:
        db.collection("predicciones").add(data)
        return jsonify({"message": "Datos guardados en Firebase correctamente"}), 200
    except Exception as e:
        print("🔥 Error al guardar en Firebase:", e)
        return jsonify({"error": str(e)}), 500


