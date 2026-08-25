import numpy as np
import os
import argparse

from part1_mlp.app.exceptions import WeightFileError
from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork

def parse_args():
    # Создаём парсер аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Обучите простую нейронную сеть MLP"
    )
    # Добавляем количество эпох обучения
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Number of training epochs",
    )
    # Добавляем скорость обучения
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    return parser.parse_args()
def main():
    # Получаем параметры из командной строки
    args = parse_args()
    # Создаём менеджер для работы с датасетом
    manager = DatasetManager()
    # Определяем путь к CSV-файлу
    dataset_path = os.path.join(
        "part1_mlp",
        "data",
        "xor.csv",
    )
    # Загружаем признаки и целевые значения из CSV
    x, y = manager.load_csv(
        dataset_path,
        "class",
    )
    # Преобразуем целевые значения в one-hot формат
    y = manager.to_one_hot(
        y,
        num_classes=2,
    )
    # Создаём нейронную сеть
    network = NeuralNetwork(
        [2, 4, 2],
        learning_rate=args.learning_rate,
    )
    # Определяем путь к файлу сохранённых весов
    weights_path = os.path.join(
        "part1_mlp",
        "weights",
        "xor_model.npy",
    )
    # Проверяем, существует ли уже сохранённая модель
    if os.path.isfile(weights_path):
        # Загружаем ранее обученные веса
        network.load_weights(weights_path)
        print("Weights loaded")
        loss_history = None
    else:
        # Обучаем нейронную сеть, если сохранённых весов нет
        loss_history = network.train(
            x,
            y,
            epochs=args.epochs,
            batch_size=2,
        )
        print(f"Loss before: {loss_history[0]}")
        print(f"Loss after: {loss_history[-1]}")
        # Сохраняем обученные веса
        network.save_weights(weights_path)
        print("Weights saved")
    # Получаем предсказания обученной или загруженной модели
    predictions = network.predict(x)
    # Оцениваем точность модели на тестовых данных
    accuracy = network.evaluate(x, y)
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