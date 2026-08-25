import numpy as np
import pandas as pd
import os

# Импортируем собственные исключения проекта.
from part1_mlp.app.exceptions import (
    DatasetFileNotFoundError,
    DatasetError,
    MismatchedDataError,
)
from part1_mlp.app.utils import validate_test_size, validate_data

class DatasetManager:
    def __init__(self):
        # Входные признаки
        self.x = None
        # Правильные ответы
        self.y = None
    def load_csv(self, file_path, target_column):
        # Проверяем, существует ли файл датасета.
        if not os.path.isfile(file_path):
            raise DatasetFileNotFoundError(
                f"Dataset file not found: {file_path}"
            )
        try:
            # Загружаем CSV-файл
            data = pd.read_csv(file_path)
            # Проверяем, что указанный столбец существует
            if target_column not in data.columns:
                raise ValueError(
                    f"Указанный столбец '{target_column}' не найденен"
                )
            # Получаем все столбцы, кроме целевого
            feature_columns = [
                column for column in data.columns
                if column != target_column
            ]
            # Преобразуем признаки в NumPy-массив
            self.x = data[feature_columns].to_numpy(dtype=float)
            # Преобразуем целевой столбец в NumPy-массив
            self.y = data[target_column].to_numpy()
            # Возвращаем подготовленные данные
            return self.x, self.y
        except DatasetError:
            raise
        except Exception as error:
            # Преобразуем неизвестную ошибку в собственное исключение
            raise DatasetError(
                f"Ошибка загрузки dataset: {error}"
            ) from error
    def normalize(self, x):
        # Вычисляем среднее значение каждого признака
        mean = np.mean(x, axis=0)
        # Вычисляем стандартное отклонение каждого признака
        std = np.std(x, axis=0)
        # Проверка деления на ноль, если какой-то признак имеет одинаковые значения.
        std = np.where(std == 0, 1, std)
        # Нормализуем признаки
        normalized_x = (x - mean) / std
        # Возвращаем нормализованные данные
        return normalized_x
    def train_test_split(self, x, y, test_size=0.2, shuffle=True):
        validate_test_size(test_size)
        validate_data(x, y)
        # Проверяем, что test_size находится в допустимом диапазоне
        if not 0 < test_size < 1:
            raise ValueError("test_size должен быть между 0 и 1")

        data_size = len(x)
        # Создаём индексы всех объектов
        indices = np.arange(data_size)
        # Перемешиваем индексы перед разделением
        if shuffle:
            np.random.shuffle(indices)
        # Определяем количество объектов для тестовой выборки
        test_count = int(data_size * test_size)
        # Получаем индексы тестовой части
        test_indices = indices[:test_count]
        # Получаем индексы обучающей части
        train_indices = indices[test_count:]

        # Формируем обучающие данные.
        x_train = x[train_indices]
        y_train = y[train_indices]

        # Формируем тестовые данные
        x_test = x[test_indices]
        y_test = y[test_indices]

        # Возвращаем четыре части датасета
        return x_train, x_test, y_train, y_test
    # Метод, который преобразует номера классов в формат, который понимает наша нейросеть
    def to_one_hot(self, y, num_classes=None):
            # Преобразуем входные классы в целочисленный NumPy-массив
            y = np.asarray(y, dtype=int)

            if num_classes is None:
                num_classes = np.max(y) + 1

            # Создаём матрицу нулей
            one_hot = np.zeros(
                (len(y), num_classes),
                dtype=float
            )
            # Для каждого объекта устанавливаем 1 в столбце соответствующего класса
            one_hot[np.arange(len(y)), y] = 1.0
            # Возвращаем one-hot представление классов
            return one_hot