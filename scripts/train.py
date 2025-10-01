"""Script para ejecutar entrenamiento desde la línea de comandos.
Ejemplo:
    python scripts/train.py --epochs 10 --batch 128 --save-dir models
"""

import argparse
import os
from src.mnist_trainer.data import DataLoader
from src.mnist_trainer.model import ModelBuilder
from src.mnist_trainer.trainer import Trainer


def parse_hidden(s: str):
    # "128,64" -> [128,64]
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--hidden", type=str, default="128,64")
    args = parser.parse_args()

    data = DataLoader(normalize=True, flatten=True)
    x_train, y_train, x_test, y_test = data.load()

    # usar una parte del test como validación
    x_val, y_val = x_test[:5000], y_test[:5000]

    input_dim = x_train.shape[1]
    hidden = parse_hidden(args.hidden)

    builder = ModelBuilder(input_dim=input_dim)
    model = builder.build(hidden_units=hidden)
    builder.compile(lr=args.lr)

    trainer = Trainer(model, save_dir=args.save_dir)
    trainer.fit(x_train, y_train, x_val=x_val, y_val=y_val,
                batch_size=args.batch, epochs=args.epochs)

    # guardamos el modelo final
    os.makedirs(args.save_dir, exist_ok=True)
    builder.save(os.path.join(args.save_dir, "final_model.h5"))


if __name__ == "__main__":
    main()
