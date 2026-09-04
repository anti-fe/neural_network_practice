import os
import queue
import threading
import time
from multiprocessing import Pool

import numpy as np

from part3_augmentation.app.augmentation import (
    process_augmentation,
)
from part3_augmentation.app.performance import (
    sequential_cpu_benchmark,
    save_performance_report,
    create_benchmark_dataset,
    sequential_cpu_benchmark,
    multiprocessing_cpu_benchmark,
)
from part3_augmentation.app.consumers import FileConsumer
from part3_augmentation.app.producer import FileProducer


# Количество потоков-потребителей
CONSUMER_COUNT = 3
# Количество процессов для аугментации
PROCESS_COUNT = 4


def threaded_read(
    data_path: str,
) -> tuple[list[tuple[str, np.ndarray]], float, int]:
    """
    Загружает файлы с помощью Producer-Consumer.

    Один Producer помещает пути файлов в очередь,
    несколько Consumer загружают данные параллельно.
    """

    # Засекаем начало обработки
    start_time = time.perf_counter()
    # Создаём очередь с путями файлов
    file_queue = queue.Queue()
    # Создаём очередь с загруженными массивами
    data_queue = queue.Queue()
    # Создаём общий счётчик
    counter = {
        "processed": 0,
    }
    # Создаём блокировку счётчика
    counter_lock = threading.Lock()
    # Создаём Producer
    producer = FileProducer(
        data_path=data_path,
        file_queue=file_queue,
    )
    # Создаём список Consumer-потоков
    consumers = []
    # Создаём и запускаем Consumer-потоки
    for consumer_id in range(
        1,
        CONSUMER_COUNT + 1,
    ):
        # Создаём очередной Consumer
        consumer = FileConsumer(
            consumer_id=consumer_id,
            file_queue=file_queue,
            data_queue=data_queue,
            counter=counter,
            counter_lock=counter_lock,
        )
        # Сохраняем поток в список
        consumers.append(consumer)
        # Запускаем поток
        consumer.start()
        
    # Запускаем Producer
    producer.start()
    # Ожидаем завершения Producer
    producer.join()
    # Ожидаем обработки всех файлов
    file_queue.join()
    # Передаём Consumer сигналы завершения
    for _ in consumers:
        file_queue.put(None)
    # Ожидаем обработки сигналов завершения
    file_queue.join()
    # Ожидаем завершения всех Consumer
    for consumer in consumers:
        consumer.join()
    # Извлекаем данные из очереди
    results = []
    # Пока очередь содержит данные
    while not data_queue.empty():
        # Получаем очередной результат
        results.append(
            data_queue.get()
        )
    # Сортируем результаты по имени файла
    results.sort(
        key=lambda item: item[0]
    )
    # Вычисляем время работы потоков
    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    return (
        results,
        elapsed_time,
        counter["processed"],
    )

def multiprocessing_augment(
    data: list[tuple[str, np.ndarray]],
) -> tuple[list[tuple[str, np.ndarray]], float]:
    """
    Выполняет аугментацию данных в нескольких процессах.

    Для передачи результатов используется callback,
    который вызывается после завершения каждого процесса.
    """

    # Засекаем начало процессной обработки.
    start_time = time.perf_counter()
    # Создаём список для результатов, полученных через callback.
    results = []
    # Создаём блокировку для безопасного добавления результатов.
    results_lock = threading.Lock()

    def collect_result(
        result: tuple[str, np.ndarray],
    ) -> None:
        """
        Callback-функция для получения результата процесса.
        """

        # Защищаем общий список результатов блокировкой
        with results_lock:
            # Добавляем результат в общий список
            results.append(result)
            # Получаем путь обработанного файла
            file_path = result[0]
            # Выводим информацию о полученном результате
            print(
                f"[Callback] Получен результат: "
                f"{file_path}"
            )
    # Создаём пул из заданного количества процессов
    with Pool(
        processes=PROCESS_COUNT
    ) as pool:
        # Отправляем каждое задание в отдельный процесс
        for item in data:
            # Запускаем обработку
            pool.apply_async(
                process_augmentation,
                args=(item,),
                callback=collect_result,
            )
        # Запрещаем добавление новых задач в пул
        pool.close()
        # Ожидаем завершения всех процессов
        pool.join()
    # Сортируем результаты по имени исходного файла
    results.sort(
        key=lambda item: item[0]
    )
    # Вычисляем общее время аугментации
    elapsed_time = (
        time.perf_counter()
        - start_time
    )
    return results, elapsed_time


def main():
    """Основная функция Part 3."""

    # Указываем каталог с исходными данными
    data_path = "part3_augmentation/data"
    # Проверяем наличие каталога
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Каталог не найден: {data_path}"
        )
    # Получаем данные через Producer-Consumer
    data, read_time, processed_count = threaded_read(
        data_path
    )
    # Создаём увеличенный набор данных специально для CPU-bound тестирования
    benchmark_data = create_benchmark_dataset(
        data,
        size=800,
    )
    # Выводим размер тестовых данных
    print(
        f"Размер benchmark-массивов: "
        f"{benchmark_data[0][1].shape}"
    )
    # Выполняем обычную многопроцессорную аугментацию.
    multiprocessing_data, multiprocessing_time = multiprocessing_augment(data)
    print(
        f"Время многопроцессорной аугментации: "
        f"{multiprocessing_time:.6f} сек."
    )
    # Выводим информацию о benchmark-наборе
    print("\n=== CPU Benchmark ===")
    print(
        f"Количество образцов: "
        f"{len(benchmark_data)}"
    )
    print(
        f"Размер одного образца: "
        f"{benchmark_data[0][1].shape}"
    )
    print(
        f"Количество процессов: "
        f"{PROCESS_COUNT}"
    )
    # Выполняем CPU-bound обработку последовательно
    benchmark_sequential_time = sequential_cpu_benchmark(
        benchmark_data
    )
    print(
        f"Последовательная обработка: "
        f"{benchmark_sequential_time:.6f} сек."
    )
    # Выполняем CPU-bound обработку через multiprocessing
    benchmark_multiprocessing_time = (
        multiprocessing_cpu_benchmark(
            benchmark_data,
            PROCESS_COUNT,
        )
    )
    print(
        f"Многопроцессорная обработка: "
        f"{benchmark_multiprocessing_time:.6f} сек."
    )
    # Объединяем результаты всех процессов
    final_data = np.concatenate(
        [
            result
            for _, result in multiprocessing_data
        ],
        axis=0,
    )
    # Создаём каталог результатов
    os.makedirs(
        "part3_augmentation/output",
        exist_ok=True,
    )
    # Указываем путь к результату
    output_path = (
        "part3_augmentation/output/"
        "multiprocessing.npy"
    )
    # Сохраняем расширенный набор данных
    np.save(
        output_path,
        final_data,
    )
    # Формируем путь к отчёту производительности
    performance_path = (
        "part3_augmentation/output/"
        "performance_report.json"
    )
   # Сохраняем полный отчёт производительности Part 3
    save_performance_report(
        output_path=performance_path,
        files_processed=processed_count,
        threaded_read_time=read_time,
        multiprocessing_time=multiprocessing_time,
        benchmark_sequential_time=benchmark_sequential_time,
        benchmark_multiprocessing_time=benchmark_multiprocessing_time,
        process_count=PROCESS_COUNT,
        benchmark_samples=len(benchmark_data),
        benchmark_shape=benchmark_data[0][1].shape,
    )
    # Выводим итоговую статистику
    print("\n=== Part 3 ===")
    print(
        f"Обработано файлов: "
        f"{processed_count}"
    )
    print(
        f"Время многопоточного чтения: "
        f"{read_time:.6f} сек."
    )
    print(
        f"Количество процессов: "
        f"{PROCESS_COUNT}"
    )
    print(
        f"Время многопроцессорной аугментации: "
        f"{multiprocessing_time:.6f} сек."
    )
    print(
        f"Размер итогового массива: "
        f"{final_data.shape}"
    )
    print(
        f"Тип данных: "
        f"{final_data.dtype}"
    )
    print(
        f"Файл сохранён: "
        f"{output_path}"
    )
    print(
        f"Отчёт производительности: "
        f"{performance_path}"
    )

if __name__ == "__main__":
    main()