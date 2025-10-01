from src.mnist_trainer.data import DataLoader
from src.mnist_trainer.model import ModelBuilder
from src.mnist_trainer.visualize import Visualizer

# Cargar datos
data = DataLoader()
x_train, y_train, x_test, y_test = data.load()

# Cargar modelo
model = ModelBuilder.load("models/final_model.h5")

# Predicciones
y_pred = model.predict(x_test[:1000])

# Matriz de confusión
Visualizer.plot_confusion(y_test[:1000], y_pred)
