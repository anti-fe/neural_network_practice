import os
import time
import numpy as np

from part3_augmentation.app.augmentation import augment_data

SUPPORTED_EXTENSIONS = {".csv"}

def find_files(data_path: str) -> list[str]:
    """
    Рекурсивно ищет поддерживаемые файлы в каталоге

    Возвращает отсортированный список путей к найденным файлам
    """

    # Список найденных файлов
    files = []
    # Рекурсивно обходим указанный каталог
    for root, _, filenames in os.walk(data_path):
        for filename in filenames:
            # Получаем расширение файла
            _, extension = os.path.splitext(filename)
            # Приводим расширение к нижнему регистру
            extension = extension.lower()
            # Добавляем только поддерживаемые файлы
            if extension in SUPPORTED_EXTENSIONS:
                files.append(
                    os.path.join(root, filename)
                )
    return sorted(files)

def load_csv(file_path: str) -> np.ndarray:
    """
    Загружает CSV-файл и возвращает нормализованный массив float32.
    """

    # Загружаем данные из CSV
    data = np.loadtxt(
        file_path,
        delimiter=",",
        dtype=np.float32,
    )
    # Находим минимальное и максимальное значения
    min_value = data.min()
    max_value = data.max()
    # Выполняем min-max нормализацию, если значения различаются
    if max_value != min_value:
        data = (
            data - min_value
        ) / (
            max_value - min_value
        )
    return data.astype(np.float32)

def sequential_prepare(
    data_path: str,
) -> tuple[np.ndarray, float]:
    """
    Последовательно загружает файлы и выполняет их аугментацию

    Возвращает итоговый массив и время обработки
    """

    # Время начала обработки
    start_time = time.perf_counter()
    # Ищем файлы для обработки
    files = find_files(data_path)
    if not files:
        raise FileNotFoundError(
            "Подходящие файлы для обработки не найдены."
        )
    # Список результатов аугментации
    results = []
    for file_path in files:
        # Загружаем CSV-файл
        data = load_csv(file_path)
        # Выполняем аугментацию
        augmented = augment_data(data)
        # Сохраняем результат обработки
        results.append(augmented)
        # Выводим информацию о текущем файле
        print(f"Обработан файл: {file_path}")

    # Объединяем результаты всех файлов
    final_data = np.concatenate(
        results,
        axis=0,
    )
    # Вычисляем время обработки
    elapsed_time = (
        time.perf_counter() - start_time
    )
    # Возвращаем данные и время обработки
    return final_data, elapsed_time

def main():
    """Основная функция последовательной обработки."""

    # Путь к тестовым данным
    data_path = "part3_augmentation/data"
    # Выполняем последовательную подготовку данных
    final_data, elapsed_time = sequential_prepare(
        data_path
    )
    # Создаём каталог для результата, если его нет
    os.makedirs(
        "part3_augmentation/output",
        exist_ok=True,
    )
    # Путь к итоговому файлу
    output_path = (
        "part3_augmentation/output/sequential.npy"
    )
    # Сохраняем результат
    np.save(output_path, final_data)
    # Выводим итоговую информацию
    print("\n=== Последовательная обработка ===")
    print(f"Размер результата: {final_data.shape}")
    print(f"Тип данных: {final_data.dtype}")
    print(f"Время обработки: {elapsed_time:.6f} сек.")
    print(f"Файл сохранён: {output_path}")

if __name__ == "__main__":
    main()