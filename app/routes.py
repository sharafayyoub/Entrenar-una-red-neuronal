from flask import Blueprint, render_template, request, jsonify, current_app
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Blueprint, request, jsonify
from .firebase_config import db


main = Blueprint('main', __name__)

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔔 /predict request received")
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No se ha enviado la imagen."})

        image_data = data['image']
        # Quitar el encabezado data:image/png;base64,...
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decodificar y abrir con PIL
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")  # grayscale
        print(f"🖼 Imagen original size: {image.size}, mode: {image.mode}")

        # Asegurar fondo negro y figura blanca:
        # Nuestro canvas pinta blanco sobre negro, si tu modelo espera fondo negro no invertimos.
        # Pero si ves predicciones raras, prueba invert=True abajo.
        image = image.resize((28, 28), Image.ANTIALIAS)

        # Opcional: centrar y normalizar contraste si es necesario
        # image = ImageOps.invert(image)  # descomenta si necesitas invertir
        arr = img_to_array(image)  # shape (28,28,1)
        print("📐 after img_to_array shape:", arr.shape, "min/max:", arr.min(), arr.max())

        # Convertir a escala 0-1
        arr = arr.astype("float32") / 255.0

        # Algunos modelos esperan: (1,28,28,1) (channels_last) o (1,1,28,28) (channels_first)
        x1 = np.expand_dims(arr, axis=0)           # (1,28,28,1)
        x2 = np.transpose(x1, (0,3,1,2))          # (1,1,28,28)

        model = current_app.config.get("MODEL", None)
        if model is None:
            print("❌ Modelo no cargado en current_app.config['MODEL']")
            return jsonify({"error": "Modelo no cargado en servidor."})

        # Intentaremos predecir con x1; si falla, lo intentamos con x2
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
        print("📊 preds shape:", preds.shape, "values:", preds)

        # Si preds es vector de tamaño 10 o (1,10)
        if preds.ndim == 2:
            preds = preds[0]
        digit = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        print(f"🔎 Resultado: {digit} con confianza {confidence:.2f}%")
        return jsonify({"digit": digit, "confidence": round(confidence, 2)})

    except Exception as e:
        print("❌ Exception en /predict:", e)
        return jsonify({"error": str(e)})
    
routes = Blueprint('routes', __name__)

@routes.route('/save', methods=['POST'])
def save_data():
    data = request.json
    if not data:
        return jsonify({"error": "No se enviaron datos"}), 400

    # Guardar datos en Firebase Firestore
    db.collection("predicciones").add(data)

    return jsonify({"message": "Datos guardados en Firebase correctamente"}), 200

