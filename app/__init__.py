from flask import Flask
from tensorflow.keras.models import load_model
import os

def create_app():
    app = Flask(__name__)

    # Intentar cargar el modelo (ajusta el nombre si tu modelo tiene otro)
    model_path_candidates = [
        os.path.join("models", "best_model.h5"),
        os.path.join("models", "final_model.h5")
    ]
    model = None
    for p in model_path_candidates:
        if os.path.exists(p):
            print(f"✅ Cargando modelo desde: {p}")
            try:
                model = load_model(p)
                print("✅ Modelo cargado correctamente")
            except Exception as e:
                print("❌ Error cargando el modelo:", e)
            break
    if model is None:
        print("⚠️ No se encontró/ pudo cargar ningún modelo en:", model_path_candidates)

    app.config["MODEL"] = model

    from .routes import main
    app.register_blueprint(main)
    return app

