import numpy as np

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
    # Обучаем сеть на XOR-данных.
    loss_history = network.train(
        x,
        y,
        epochs=5000,
        batch_size=2,
    )
    # Получаем предсказания обученной сети
    predictions = network.predict(x)
    # Оцениваем точность модели
    accuracy = network.evaluate(x, y)
    # Выводим начальную ошибку
    print(f"Loss before: {loss_history[0]}")
    # Выводим конечную ошибку
    print(f"Loss after: {loss_history[-1]}")
    # Выводим предсказанные классы
    print(f"Predictions: {predictions}")
    # Выводим точность модели
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Запускаем основной сценарий программы
    main()