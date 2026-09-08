import os

from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork


def console_menu():
    """Функция для отображения главного меню в консоли"""
    # Выводим пункты главного меню.
    print("\n=== Neural Network MLP ===")
    print("1. Создать сеть")
    print("2. Загрузить данные")
    print("3. Обучить сеть")
    print("4. Выполнить предсказание")
    print("5. Сохранить веса")
    print("6. Загрузить веса")
    print("7. Выход")

    # Получаем выбор пользователя.
    return input("Выберите пункт: ")
def main():
    """Запускает консольное меню и управляет работой нейронной сети."""

    # Создаём менеджер для работы с датасетом.
    manager = DatasetManager()

    # Переменные для хранения данных и нейронной сети.
    x = None
    y = None
    network = None

    # Путь к XOR-датасету.
    dataset_path = os.path.join(
        "part1_mlp",
        "data",
        "xor.csv",
    )

    # Путь к файлу сохранённых весов.
    weights_path = os.path.join(
        "part1_mlp",
        "weights",
        "xor_model.npy",
    )

    # Запускаем главное меню
    while True:
        choice = console_menu()

        # Создание нейронной сети
        if choice == "1":
            try:
                # Запрашиваем архитектуру сети
                layer_sizes_input = input(
                    "Введите размеры слоёв через пробел "
                    "(например: 2 4 2): "
                )

                # Преобразуем ввод в список целых чисел
                layer_sizes = [
                    int(size)
                    for size in layer_sizes_input.split()
                ]

                # Выбираем функцию активации
                print("\nВыберите функцию активации:")
                print("1. ReLU")
                print("2. Sigmoid")

                activation_choice = input("Ваш выбор: ")

                if activation_choice == "1":
                    activation = "relu"
                elif activation_choice == "2":
                    activation = "sigmoid"
                else:
                    print("Некорректный выбор.")
                    continue

                # Запрашиваем скорость обучения
                learning_rate = float(
                    input("Введите скорость обучения: ")
                )

                # Создаём нейронную сеть.
                network = NeuralNetwork(
                    layer_sizes,
                    learning_rate=learning_rate,
                    activation=activation,
                )

                print(
                    f"Сеть создана: {layer_sizes}, "
                    f"активация: {activation}"
                )

            except ValueError as error:
                print(f"Ошибка ввода: {error}")

        # Загрузка данных
        elif choice == "2":
            try:
                # Позволяем использовать стандартный XOR-файл или указать собственный путь
                user_path = input(
                    f"Путь к CSV "
                    f"(Enter — {dataset_path}): "
                )

                if user_path.strip():
                    dataset_path = user_path

                # Загружаем данные из CSV.
                x, y = manager.load_csv(
                    dataset_path,
                    "class",
                )
                # Нормализуем входные признаки.
                x = manager.normalize(x)
                # Преобразуем классы в one-hot формат.
                y = manager.to_one_hot(y, num_classes=2)

                print("Данные успешно загружены.")
                print(f"Количество объектов: {len(x)}")

            except Exception as error:
                print(f"Ошибка загрузки данных: {error}")

        # Обучение сети
        elif choice == "3":
            if network is None:
                print("Сначала создайте сеть.")
                continue
            if x is None or y is None:
                print("Сначала загрузите данные.")
                continue
            try:
                # Запрашиваем параметры обучения
                epochs = int(input("Количество эпох: "))
                batch_size = int(input("Размер batch: "))

                # Обучаем сеть на полном XOR-наборе
                loss_history = network.train(
                    x,
                    y,
                    epochs=epochs,
                    batch_size=batch_size,
                )

                print(f"Loss before: {loss_history[0]}")
                print(f"Loss after: {loss_history[-1]}")

                print("Обучение завершено.")

            except ValueError as error:
                print(f"Ошибка ввода: {error}")

        # Предсказание
        elif choice == "4":
            if network is None:
                print("Сначала создайте сеть.")
                continue
            if x is None or y is None:
                print("Сначала загрузите данные.")
                continue
            try:
                # Получаем предсказания для всего набора
                predictions = network.predict(x)

                # Вычисляем точность
                accuracy = network.evaluate(x, y)

                print(f"Predictions: {predictions}")
                print(f"Accuracy: {accuracy * 100:.2f}%")
            except Exception as error:
                print(f"Ошибка предсказания: {error}")

        # Сохранение весов
        elif choice == "5":
            if network is None:
                print("Сначала создайте сеть.")
                continue
            try:
                # Сохраняем веса модели
                network.save_weights(weights_path)
                print(f"Weights saved: {weights_path}")
            except Exception as error:
                print(f"Ошибка сохранения весов: {error}")

        # Загрузка весов
        elif choice == "6":
            if network is None:
                print("Сначала создайте сеть.")
                continue
            try:
                # Загружаем ранее сохранённые веса
                network.load_weights(weights_path)

                print(f"Weights loaded: {weights_path}")

            except Exception as error:
                print(f"Ошибка загрузки весов: {error}")

        # Выход из программы
        elif choice == "7":
            print("Выход из программы.")
            break
        else:
            print("Некорректный пункт меню.")

if __name__ == "__main__":
    # Запускаем основной сценарий программы
    main()