import json
import time
from typing import Any
import numpy as np
from part3_augmentation.app.augmentation import (
    augment_data,
)

def sequential_augment(
    data: list[tuple[str, np.ndarray]],
) -> tuple[list[tuple[str, np.ndarray]], float]:
    """
    Выполняет аугментацию последовательно,
    используя один основной поток.
    """

    # Засекаем начало последовательной обработки
    start_time = time.perf_counter()
    # Создаём список результатов
    results = []

    for file_path, array in data:
        # Выполняем аугментацию текущего массива
        augmented = augment_data(array)
        # Сохраняем результат
        results.append(
            (
                file_path,
                augmented,
            )
        )
    # Вычисляем время выполнения
    elapsed_time = (
        time.perf_counter()
        - start_time
    )
    return results, elapsed_time

def calculate_speedup(
    sequential_time: float,
    multiprocessing_time: float,
) -> float:
    """
    Рассчитывает ускорение многопроцессорной обработки
    относительно последовательной.
    """

    # Проверяем, что последовательное время не равно нулю
    if sequential_time == 0:
        return 0.0
    # Рассчитываем коэффициент ускорения
    return (
        sequential_time
        / multiprocessing_time
    )

def save_performance_report(
    output_path: str,
    files_processed: int,
    sequential_time: float,
    multiprocessing_time: float,
    process_count: int,
) -> None:
    """
    Сохраняет результаты сравнения в JSON-файл.
    """

    # Рассчитываем коэффициент ускорения
    speedup = calculate_speedup(
        sequential_time,
        multiprocessing_time,
    )
    # Формируем структуру отчёта
    report: dict[str, Any] = {
        "files_processed": files_processed,
        "sequential_time": round(
            sequential_time,
            6,
        ),
        "multiprocessing_time": round(
            multiprocessing_time,
            6,
        ),
        "speedup": round(
            speedup,
            6,
        ),
        "process_count": process_count,
    }
    # Открываем JSON-файл для записи
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        # Сохраняем отчёт с красивым форматированием
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )
    print(
        f"Отчёт производительности сохранён: "
        f"{output_path}"
    )