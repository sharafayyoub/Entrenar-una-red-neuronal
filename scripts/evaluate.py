import argparse
from src.mnist_trainer.data import DataLoader
from src.mnist_trainer.model import ModelBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ruta al .h5 guardado")
    args = parser.parse_args()

    data = DataLoader(normalize=True, flatten=True)
    _, _, x_test, y_test = data.load()

    model = ModelBuilder.load(args.model)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}  -  Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
