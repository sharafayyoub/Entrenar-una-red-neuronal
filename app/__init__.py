from flask import Flask
from tensorflow.keras.models import load_model
import os

model = None

def create_app():
    app = Flask(__name__)
    
    from .routes import main
    app.register_blueprint(main)

    global model
    model_path = os.path.join("models", "best_model.h5")
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado en: {model_path}")
    else:
        print(f"✅ Cargando modelo desde: {model_path}")
        model = load_model(model_path)
        print("✅ Modelo cargado correctamente")

    return app
