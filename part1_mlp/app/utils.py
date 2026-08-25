import numpy as np


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