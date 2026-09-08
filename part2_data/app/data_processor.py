import os
import time
import numpy as np


def find_data_files(data_path, extensions):
    """Рекурсивно ищет файлы с указанными расширениями"""

    # Проверяем существование каталога
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Каталог с данными не найден: {data_path}"
        )
    # Приводим расширения к единому формату
    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }

    # Создаём список найденных файлов
    data_files = []

    # Рекурсивно обходим каталог и его подкаталоги
    for root, _, files in os.walk(data_path):
        for filename in files:
            # Получаем расширение файла
            extension = os.path.splitext(filename)[1].lower()
            # Проверяем, поддерживается ли это расширение
            if extension in normalized_extensions:
                data_files.append(
                    os.path.join(root, filename)
                )

    # Сортируем файлы для стабильного порядка обработки
    data_files.sort()
    return data_files


def load_csv(file_path):
    """Загружает CSV-файл в NumPy-массив"""

    # Загружаем числовую таблицу из CSV
    data = np.loadtxt(
        file_path,
        delimiter=",",
    )
    # Приводим данные к float32
    return np.asarray(
        data,
        dtype=np.float32,
    )

def load_image(file_path):
    """Загружает изображение через Pillow"""

    # Импортируем Pillow только при работе с изображениями
    from PIL import Image

    # Открываем изображение
    with Image.open(file_path) as image:
        # Преобразуем изображение в RGB
        image = image.convert("RGB")
        # Преобразуем изображение в NumPy-массив
        data = np.asarray(
            image,
            dtype=np.float32,
        )
    # Нормализуем значения пикселей от 0 до 1
    return data / 255.0

def load_file(file_path):
    """Загружает файл в зависимости от его расширения"""

    # Получаем расширение файла
    extension = os.path.splitext(file_path)[1].lower()
    # Обрабатываем CSV
    if extension == ".csv":
        return load_csv(file_path)

    # Обрабатываем изображения
    if extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
    }:
        return load_image(file_path)

    # Сообщаем об неизвестном формате
    raise ValueError(
        f"Неподдерживаемый формат файла: {file_path}"
    )

def normalize_csv(data):
    """Нормализует числовые данные CSV"""

    # Приводим массив к float32
    data = np.asarray(
        data,
        dtype=np.float32,
    )
    # Если массив пустой, возвращаем его без изменений
    if data.size == 0:
        return data
    # Находим минимальное и максимальное значение
    data_min = data.min()
    data_max = data.max()
    # Если все значения одинаковые, нормализация не требуется
    if data_max == data_min:
        return np.zeros_like(data)
    # Выполняем min-max нормализацию в диапазон 0..1.
    return (data - data_min) / (data_max - data_min)

def normalize_data(data, file_path):
    """Нормализует данные с учётом типа файла"""

    # Получаем расширение файла
    extension = os.path.splitext(file_path)[1].lower()
    # Изображения уже нормализуются через деление на 255
    if extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
    }:
        return np.asarray(
            data,
            dtype=np.float32,
        )
    # Для CSV используем min-max нормализацию
    if extension == ".csv":
        return normalize_csv(data)
    # Для неизвестного типа оставляем понятную ошибку
    raise ValueError(
        f"Невозможно нормализовать файл: {file_path}"
    )

def prepare_data(data_path, extensions):
    """Загружает файлы и приводит их к единому формату"""

    # Запоминаем время начала обработки
    start_time = time.perf_counter()
    # Ищем подходящие файлы
    files = find_data_files(
        data_path,
        extensions,
    )
    # Проверяем наличие файлов
    if not files:
        raise FileNotFoundError(
            f"В каталоге {data_path} "
            f"не найдено файлов с расширениями: {extensions}"
        )
    # Сохраняем обработанные массивы
    arrays = []
    # Сохраняем исходные размеры файлов
    original_shapes = []

    # Обрабатываем каждый файл
    for file_path in files:
        data = load_file(file_path)
        # Запоминаем исходный размер
        original_shapes.append(
            {
                "file": file_path,
                "shape": data.shape,
            }
        )
        # Нормализуем данные
        data = normalize_data(data, file_path)
        # Преобразуем данные в одномерный вектор
        data = data.reshape(-1)
        # Добавляем в общий список
        arrays.append(data)
    # Находим максимальное количество элементов среди файлов
    max_size = max(
        array.size
        for array in arrays
    )
    # Создаём список массивов одинаковой длины
    unified_arrays = []
    # Приводим каждый образец к одинаковому размеру
    for array in arrays:
        # Вычисляем, сколько элементов необходимо добавить
        padding_size = max_size - array.size
        # Добавляем нулевые значения справа
        padded_array = np.pad(
            array,
            (0, padding_size),
            mode="constant",
            constant_values=0,
        )
        # Добавляем подготовленный образец
        unified_arrays.append(
            padded_array.astype(np.float32)
        )
    # Объединяем все образцы в двумерный массив
    combined = np.vstack(unified_arrays).astype(np.float32)
    # Вычисляем время обработки
    elapsed_time = time.perf_counter() - start_time
    
    # Формируем статистику
    statistics = {
        "file_count": len(files),
        "shape": combined.shape,
        "dtype": str(combined.dtype),
        "processing_time": elapsed_time,
        "original_shapes": original_shapes,
    }
    return combined, statistics