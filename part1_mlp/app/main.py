import numpy as np
import os

from part1_mlp.app.exceptions import WeightFileError
from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork


def main():
    # Создаём небольшой XOR-датасет для проверки всей системы
    x = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    # Создаём правильные ответы в one-hot формате
    y = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    # Создаём нейронную сеть
    network = NeuralNetwork(
        [2, 4, 2],
        learning_rate=0.1,
    )
    # Определяем путь к файлу сохранённых весов
    weights_path = os.path.join(
        "part1_mlp",
        "weights",
        "xor_model.npy",
    )
    # Обучаем сеть на XOR-данных.
    loss_history = network.train(
        x,
        y,
        epochs=5000,
        batch_size=2,
    )
    # Сохраняем обученную модель в файл
    network.save_weights(weights_path)
    print(f"Weights saved: {weights_path}")

    predictions = network.predict(x)
    accuracy = network.evaluate(x, y)
    print(f"Loss before: {loss_history[0]}")
    print(f"Loss after: {loss_history[-1]}")
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
def save_weights(self, file_path):
    # Сохраняем веса модели в файл
    try:
        np.save(
            file_path,
            {
                "weights": self.weights,
                "biases": self.biases,
            },
            allow_pickle=True,
        )

    except Exception as error:
        raise WeightFileError(
            f"Ошибка сохранения weights: {error}"
        ) from error

if __name__ == "__main__":
    # Запускаем основной сценарий программы
    main()