import numpy as np
import json

def validate_data(x, y):
    # Проверяем, что входные данные являются NumPy-массивами.
    if not isinstance(x, np.ndarray):
        raise TypeError("x должен быть NumPy массивом")

    # Проверяем, что правильные ответы являются NumPy-массивом.
    if not isinstance(y, np.ndarray):
        raise TypeError("y должен быть NumPy массивом")

    # Проверяем, что количество объектов совпадает.
    if len(x) != len(y):
        raise ValueError("x и y должны содержать одно и то же количество образцов")

    # Проверяем, что входные данные не пустые.
    if x.size == 0:
        raise ValueError("x должен быть непустым")
    # Проверяем, что целевые значения не пустые.
    if y.size == 0:
        raise ValueError("y не должен быть пустым")

    # Возвращаем True, если все проверки пройдены.
    return True
def save_loss_history(loss_history, file_path):
    # Преобразуем значения NumPy в обычные числа
    history = [float(loss) for loss in loss_history]

    # Открываем файл для записи
    with open(file_path, "w", encoding="utf-8") as file:
        # Сохраняем историю обучения в JSON.
        json.dump(history, file, indent=4)
def load_loss_history(file_path):
    # Открываем файл с историей обучения
    with open(file_path, "r", encoding="utf-8") as file:
        # Загружаем историю из JSON
        history = json.load(file)

    # Возвращаем загруженную историю.
    return history