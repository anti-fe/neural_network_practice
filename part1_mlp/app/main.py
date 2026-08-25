import os
import argparse

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
    # Нормализуем входные признаки перед обучением
    x = manager.normalize(x)
    # Преобразуем целевые значения в one-hot формат
    y = manager.to_one_hot(
        y,
        num_classes=2,
    )
    # Разделяем датасет на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = manager.train_test_split(
        x,
        y,
        test_size=0.25,
        shuffle=True,
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
        # Обучаем нейронную сеть только на обучающей выборке
        loss_history = network.train(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=2,
        )
        print(f"Loss before: {loss_history[0]}")
        print(f"Loss after: {loss_history[-1]}")
        # Сохраняем обученные веса
        network.save_weights(weights_path)
        print("Weights saved")
    # Получаем предсказания на тестовой выборке
    predictions = network.predict(x_test)
    # Оцениваем модель на данных, которые не использовались при обучении
    accuracy = network.evaluate(x_test, y_test)
    # Выводим предсказания и точность
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Запускаем основной сценарий программы
    main()