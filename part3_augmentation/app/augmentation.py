import numpy as np


def augment_data(data: np.ndarray) -> np.ndarray:
    """
    Выполняет аугментацию массива.

    Функция создаёт несколько вариантов исходных данных:
    поворот, отражение и добавление случайного шума.
    """

    # Создаём список для хранения аугментированных вариантов
    augmented_data = []
    # Добавляем исходные данные
    augmented_data.append(data)
    # Выполняем поворот на 90 градусов, в случае, если данные являются изображением с двумя или тремя измерениями
    if data.ndim >= 2:
        rotated = np.rot90(data)
        augmented_data.append(rotated)
        # Выполняем горизонтальное отражение
        flipped = np.flip(data, axis=1)
        augmented_data.append(flipped)

    # Создаём случайный шум
    noise = np.random.normal(
        loc=0.0,
        scale=0.02,
        size=data.shape,
    )
    # Добавляем шум к исходным данным
    noisy = data + noise
    # Ограничиваем значения диапазоном от 0 до 1
    noisy = np.clip(noisy, 0.0, 1.0)
    # Добавляем вариант с шумом
    augmented_data.append(noisy.astype(np.float32))
    # Возвращаем все варианты аугментации
    return np.array(
        augmented_data,
        dtype=np.float32,
    )
def process_augmentation(
    item: tuple[str, np.ndarray],
) -> tuple[str, np.ndarray]:
    """
    Worker-функция для отдельного процесса.

    Получает путь файла и его массив,
    выполняет CPU-bound аугментацию
    и возвращает результат.
    """

    # Разбираем входной объект на путь и данные
    file_path, data = item
    # Выполняем аугментацию
    augmented = augment_data(data)
    
    return file_path, augmented

if __name__ == "__main__":
    # Создаём тестовое изображение размером 3 × 3.
    test_data = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )

    # Выполняем аугментацию тестовых данных.
    result = augment_data(test_data)

    # Выводим исходную форму массива.
    print(f"Исходная форма: {test_data.shape}")

    # Выводим форму результата.
    print(f"Форма результата: {result.shape}")

    # Выводим количество полученных вариантов.
    print(f"Количество вариантов: {len(result)}")