from typing import Optional, List
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class ModelBuilder:
    """Construye y compila modelos Keras de forma programática."""

    def __init__(self, input_dim: int, num_classes: int = 10):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model: Optional[Sequential] = None

    def build(self, hidden_units: List[int] = [784, 64], activation: str = "relu") -> Sequential:
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        for units in hidden_units:
            model.add(Dense(units, activation=activation))
        model.add(Dense(self.num_classes, activation="softmax"))
        self.model = model
        return model

    def compile(self, lr: float = 1e-3):
        if self.model is None:
            raise RuntimeError("Build the model before calling compile()")
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(path)

    @staticmethod
    def load(path: str):
        return load_model(path)
