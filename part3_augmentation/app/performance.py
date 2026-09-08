import json
import time
from typing import Any
import numpy as np

def sequential_cpu_benchmark(
    data: list[tuple[str, np.ndarray]],
) -> float:
    """
    Последовательно выполняет CPU-bound benchmark.
    """

    # Засекаем начало выполнения
    start_time = time.perf_counter()
    for item in data:
        # Выполняем CPU-bound вычисления
        cpu_bound_augmentation(item)
    # Возвращаем затраченное время
    return (
        time.perf_counter()
        - start_time
    )

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
def multiprocessing_cpu_benchmark(
    data: list[tuple[str, np.ndarray]],
    process_count: int,
) -> float:
    """
    Выполняет CPU-bound benchmark через multiprocessing.

    Для передачи результатов используется
    apply_async() и callback.
    """

    # Импортируем Pool для создания процессов
    from multiprocessing import Pool
    # Засекаем начало выполнения
    start_time = time.perf_counter()
    # Создаём список для результатов callback
    results = []
    # Функция вызывается после завершения каждого процесса
    def collect_result(
        result: tuple[str, np.ndarray],
    ) -> None:
        """
        Получает результат завершившегося процесса.
        """

        # Добавляем результат в список
        results.append(result)
        # Выводим информацию о завершившейся задаче
        print(
            f"[Benchmark callback] "
            f"Завершено: {result[0]}"
        )
    # Создаём пул процессов
    with Pool(
        processes=process_count
    ) as pool:
        # Создаём асинхронные задачи
        async_results = []
        # Отправляем каждый образец в процесс
        for item in data:
            # Запускаем CPU-bound обработку
            async_result = pool.apply_async(
                cpu_bound_augmentation,
                args=(item,),
                callback=collect_result,
            )
            # Сохраняем объект асинхронного результата
            async_results.append(
                async_result
            )
        # Ожидаем завершения всех задач
        for async_result in async_results:
            # Получаем результат задачи
            async_result.get()
        # Закрываем пул для новых задач
        pool.close()
        # Ожидаем завершения всех процессов
        pool.join()
    # Вычисляем общее время выполнения
    elapsed_time = (
        time.perf_counter()
        - start_time
    )
    return elapsed_time
# Сохраняем результаты производительности в JSON-файл
def save_performance_report(
    output_path: str,
    files_processed: int,
    threaded_read_time: float,
    multiprocessing_time: float,
    benchmark_sequential_time: float,
    benchmark_multiprocessing_time: float,
    process_count: int,
    benchmark_samples: int,
    benchmark_shape: tuple[int, int],
) -> None:
    """
    Сохраняет результаты производительности Part 3.

    В отчёт попадают:
    - количество обработанных файлов;
    - время многопоточного чтения;
    - время многопроцессорной аугментации;
    - результаты CPU benchmark;
    - коэффициент ускорения multiprocessing.
    """

    # Рассчитываем ускорение CPU-bound обработки
    benchmark_speedup = calculate_speedup(
        benchmark_sequential_time,
        benchmark_multiprocessing_time,
    )
    # Формируем структуру итогового отчёта
    report: dict[str, Any] = {
        # Количество обработанных исходных файлов
        "files_processed": files_processed,
        # Время многопоточного чтения файлов
        "threaded_read_time": round(
            threaded_read_time,
            6,
        ),
        # Результаты многопроцессорной аугментации
        "augmentation": {
            "multiprocessing_time": round(
                multiprocessing_time,
                6,
            ),
            "process_count": process_count,
        },
        # Результаты отдельного CPU-bound benchmark
        "cpu_benchmark": {
            # Количество тестовых образцов
            "samples": benchmark_samples,
            # Размер одного массива
            "array_shape": list(benchmark_shape),
            # Время последовательной обработки
            "sequential_time": round(
                benchmark_sequential_time,
                6,
            ),
            # Время многопроцессорной обработки
            "multiprocessing_time": round(
                benchmark_multiprocessing_time,
                6,
            ),
            # Коэффициент ускорения
            "speedup": round(
                benchmark_speedup,
                6,
            ),
            # Количество процессов
            "process_count": process_count,
        },
    }
    # Открываем JSON-файл для записи
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        # Сохраняем JSON
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
def create_benchmark_dataset(
    source_data: list[tuple[str, np.ndarray]],
    copies: int = 100,
    size: int = 800,
) -> list[tuple[str, np.ndarray]]:
    """
    Создаёт отдельный набор данных для benchmark.

    Исходные файлы проекта не изменяются.
    Из имеющихся массивов создаётся большое количество
    одинаковых по размеру тестовых образцов.
    """

    # Создаём список benchmark-данных
    benchmark_data = []

    # Перебираем необходимое количество копий
    for index in range(copies):
        # Выбираем исходный массив по кругу
        source_path, source_array = source_data[
            index % len(source_data)
        ]
        # Изменяем размер массива для увеличения нагрузки
        benchmark_array = np.resize(
            source_array,
            (size, size),
        )
        # Приводим данные к float32
        benchmark_array = benchmark_array.astype(
            np.float32
        )
        # Формируем виртуальное имя benchmark-файла
        benchmark_path = (
            f"benchmark_{index + 1}.npy"
        )
        # Добавляем тестовый образец
        benchmark_data.append(
            (
                benchmark_path,
                benchmark_array,
            )
        )
    return benchmark_data
def cpu_bound_augmentation(
    item: tuple[str, np.ndarray],
) -> tuple[str, np.ndarray]:
    """
    Выполняет CPU-bound вычисления над одним массивом.

    Функция используется только для сравнения
    последовательной и многопроцессорной обработки.
    """

    # Получаем имя образца и массив
    file_path, data = item
    # Создаём копию массива
    result = data.copy()
    # Выполняем вычислительно затратные операции
    for _ in range(10):
        # Вычисляем квадратный корень
        result = np.sqrt(
            result + 1e-6
        )
        # Выполняем возведение в степень
        result = np.power(
            result,
            1.5,
        )
        # Выполняем синус
        result = np.sin(
            result
        )
        # Выполняем косинус
        result = np.cos(
            result
        )
    return file_path, result