from src.mnist_trainer.data import DataLoader
from src.mnist_trainer.model import ModelBuilder
from src.mnist_trainer.visualize import Visualizer

def main():
    # Cargar datos
    data = DataLoader()
    _, _, x_test, y_test = data.load()

    # Cargar modelo entrenado
    model = ModelBuilder.load("models/final_model.h5")

    # Predicciones
    y_pred = model.predict(x_test[:1000])

    # Matriz de confusión
    Visualizer.plot_confusion(y_test[:1000], y_pred)

if __name__ == "__main__":
    main()
