import numpy as np
import json

from part1_mlp.app.exceptions import MismatchedDataError
from part1_mlp.app.exceptions import InvalidInputError

def validate_data(x, y):
    """Функция для проверки корректности входных данных"""
    # Проверяем, что входные данные являются NumPy-массивами.
    if not isinstance(x, np.ndarray):
        raise TypeError("x должен быть NumPy массивом")

    # Проверяем, что правильные ответы являются NumPy-массивом.
    if not isinstance(y, np.ndarray):
        raise TypeError("y должен быть NumPy массивом")

    # Проверяем, что количество объектов совпадает.
    if len(x) != len(y):
        raise MismatchedDataError("x и y должны содержать одно и то же количество образцов")

    # Проверяем, что входные данные не пустые.
    if x.size == 0:
        raise ValueError("x должен быть непустым")
    # Проверяем, что целевые значения не пустые.
    if y.size == 0:
        raise ValueError("y не должен быть пустым")

    # Возвращаем True, если все проверки пройдены.
    return True
def save_loss_history(loss_history, file_path):
    """Функция для сохранения истории обучения в JSON-файл"""
    # Преобразуем значения NumPy в обычные числа
    history = [float(loss) for loss in loss_history]

    # Открываем файл для записи
    with open(file_path, "w", encoding="utf-8") as file:
        # Сохраняем историю обучения в JSON.
        json.dump(history, file, indent=4)
def load_loss_history(file_path):
    """Функция для загрузки истории обучения из JSON-файла"""
    # Открываем файл с историей обучения
    with open(file_path, "r", encoding="utf-8") as file:
        # Загружаем историю из JSON
        history = json.load(file)

    # Возвращаем загруженную историю.
    return history
def validate_test_size(test_size):
    """Функция для проверки корректности параметра test_size"""
    if not isinstance(test_size, (int, float)):
        raise InvalidInputError(
            "test_size должен быть числом"
        )
    if not 0 < test_size < 1:
        raise InvalidInputError(
            "test_size должен находиться между 0 и 1"
        )
def validate_data(x, y):
    """Функция для проверки корректности входных данных"""
    # Проверяем, что количество объектов X и Y совпадает
    if len(x) != len(y):
        raise MismatchedDataError(
            "x и y должны содержать одно и то же количество образцов"
        )