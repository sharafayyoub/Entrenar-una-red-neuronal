from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class DataLoader:
    """Carga y preprocesa MNIST.

    - normalize: divide por 255.0
    - flatten: aplana imágenes a vectores (n, 784). 
      Si False devuelve (n,28,28)
    """

    def __init__(self, normalize: bool = True, flatten: bool = True):
        self.normalize = normalize
        self.flatten = flatten
        self.input_shape = None

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (trainX, trainY), (testX, testY) = mnist.load_data()

        if self.normalize:
            trainX = trainX.astype("float32") / 255.0
            testX = testX.astype("float32") / 255.0

        if self.flatten:
            trainX = trainX.reshape((trainX.shape[0], -1))
            testX = testX.reshape((testX.shape[0], -1))
            self.input_shape = trainX.shape[1]
        else:
            self.input_shape = trainX.shape[1:]

        trainY_cat = to_categorical(trainY)
        testY_cat = to_categorical(testY)

        return trainX, trainY_cat, testX, testY_cat

    def sample(self, x: np.ndarray, y: np.ndarray, index: int = 0):
        """Devuelve (imagen, etiqueta) cruda para index (útil para display)."""
        return x[index], y[index]
