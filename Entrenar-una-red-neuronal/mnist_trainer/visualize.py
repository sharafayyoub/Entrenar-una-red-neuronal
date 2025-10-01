import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Visualizer:

    @staticmethod
    def plot_sample(x: np.ndarray, y: np.ndarray = None, index: int = 0, flatten: bool = True):
        if flatten:
            img = x[index].reshape(28, 28)
        else:
            img = x[index]
        plt.imshow(img, cmap="gray")
        if y is not None:
            label = np.argmax(y[index]) if y.ndim > 1 else y[index]
            plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_history(history):
        """history: objeto devuelto por model.fit()"""
        plt.figure()
        plt.plot(history.history["loss"], label="loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        if "accuracy" in history.history:
            plt.figure()
            plt.plot(history.history["accuracy"], label="accuracy")
            if "val_accuracy" in history.history:
                plt.plot(history.history["val_accuracy"], label="val_accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    @staticmethod
    def plot_confusion(y_true, y_pred):
        y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()
