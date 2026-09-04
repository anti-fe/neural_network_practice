import queue
import threading
import numpy as np

class FileConsumer(threading.Thread):
    """
    Получает пути файлов из очереди, загружает данные
    и помещает подготовленные NumPy-массивы в другую очередь
    """

    def __init__(
        self,
        consumer_id: int,
        file_queue: queue.Queue,
        data_queue: queue.Queue,
        counter: dict,
        counter_lock: threading.Lock,
    ):
        """
        Инициализирует поток-потребитель

        Аргументы:
            consumer_id: Идентификатор потока.
            file_queue: Очередь с путями к файлам.
            data_queue: Очередь для подготовленных массивов.
            counter: Общий счётчик обработанных файлов.
            counter_lock: Блокировка для безопасного доступа к счётчику.
        """

        # Инициализируем базовый класс потока
        super().__init__()
        self.consumer_id = consumer_id
        self.file_queue = file_queue
        self.data_queue = data_queue
        self.counter = counter

        # Блокировка общего счётчика
        self.counter_lock = counter_lock

    @staticmethod
    def load_csv(file_path: str) -> np.ndarray:
        """
        Загружает CSV и выполняет min-max нормализацию
        """

        # Загружаем CSV в массив float32
        data = np.loadtxt(
            file_path,
            delimiter=",",
            dtype=np.float32,
        )
        # Определяем минимальное и максимальное значения
        min_value = data.min()
        max_value = data.max()
        # Нормализуем данные, если значения не одинаковые
        if max_value != min_value:
            data = (data - min_value) / (max_value - min_value)
        return data.astype(np.float32)

    def run(self) -> None:
        """
        Последовательно получает задания из очереди

        None используется как специальный сигнал завершения работы
        """
        print(
            f"[Consumer {self.consumer_id}] "
            f"Запущен."
        )
        while True:
            # Получаем следующий элемент из очереди
            file_path = self.file_queue.get()
            try:
                # None означает, что Producer завершил работу
                if file_path is None:
                    print(
                        f"[Consumer {self.consumer_id}] "
                        f"Завершён."
                    )
                    return
                # Загружаем и нормализуем файл
                data = self.load_csv(file_path)
                # Передаём подготовленный массив дальше
                self.data_queue.put(
                    (file_path, data)
                )
                # Защищаем изменение общего счётчика блокировкой
                with self.counter_lock:
                    self.counter["processed"] += 1
                    processed = self.counter["processed"]
                # Выводим информацию о работе потока
                print(
                    f"[Consumer {self.consumer_id}] "
                    f"Обработан: {file_path}. "
                    f"Всего: {processed}"
                )
            finally:
                # Сообщаем очереди о завершении обработки задания
                self.file_queue.task_done()