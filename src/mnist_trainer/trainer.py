from typing import Optional
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class Trainer:
    """Encapsula el proceso de entrenamiento: fit, evaluate, predict, callbacks."""

    def __init__(self, model, save_dir: str = "models"):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def fit(self,
            x_train,
            y_train,
            x_val=None,
            y_val=None,
            batch_size: int = 128,
            epochs: int = 10):

        callbacks = []
        checkpoint_path = os.path.join(self.save_dir, "best_model.h5")
        callbacks.append(
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor=("val_loss" if x_val is not None else "loss"))
        )

        if x_val is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True))

            history = self.model.fit(
                x_train, y_train, validation_data=(x_val, y_val),
                batch_size=batch_size, epochs=epochs, callbacks=callbacks
            )
        else:
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size, epochs=epochs, callbacks=callbacks
            )

        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
