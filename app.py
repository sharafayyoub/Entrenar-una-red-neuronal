from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import cv2
from werkzeug.utils import secure_filename

# Configuración de Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Cargar modelo entrenado
MODEL_PATH = "models/final_model.h5"
model = load_model(MODEL_PATH)

# Cargar dataset MNIST (para mostrar ejemplos)
(_, _), (x_test, y_test) = mnist.load_data()

def preprocess_image(image_path):
    """Preprocesa la imagen para que el modelo pueda predecir"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))        # Redimensionar
    img = img.astype("float32") / 255.0    # Normalizar
    img = img.reshape(1, 28, 28, 1)        # Dar forma (batch, 28,28,1)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_file = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                uploaded_file = filepath

                # Predecir
                img = preprocess_image(filepath)
                pred = model.predict(img)
                prediction = np.argmax(pred)

    # Seleccionar 10 ejemplos de MNIST para mostrar
    indices = np.random.choice(len(x_test), 10, replace=False)
    examples = [(x_test[i], y_test[i]) for i in indices]

    return render_template("index.html", prediction=prediction, uploaded_file=uploaded_file, examples=examples)

if __name__ == "__main__":
    app.run(debug=True)
