import queue
import threading
import os


SUPPORTED_EXTENSIONS = {".csv"}

class FileProducer(threading.Thread):
    """
    Поток-производитель.

    Рекурсивно сканирует каталог и помещает пути
    найденных файлов в общую очередь.
    """

    def __init__(
        self,
        data_path: str,
        file_queue: queue.Queue,
    ):
        # Инициализируем базовый класс Thread
        super().__init__()
        # Сохраняем путь к каталогу
        self.data_path = data_path
        # Сохраняем очередь для передачи файлов потребителям
        self.file_queue = file_queue
    def run(self) -> None:
        """Сканирует каталог и добавляет найденные файлы в очередь."""
        # Выводим информацию о начале работы Producer
        print(
            f"[Producer] Начало сканирования: "
            f"{self.data_path}"
        )
        # Рекурсивно обходим каталог
        for root, _, filenames in os.walk(self.data_path):
            for filename in sorted(filenames):
                # Получаем расширение файла
                _, extension = os.path.splitext(filename)
                extension = extension.lower()
                # Пропускаем неподдерживаемые форматы
                if extension not in SUPPORTED_EXTENSIONS:
                    continue
                # Формируем полный путь к файлу
                file_path = os.path.join(
                    root,
                    filename,
                )
                # Помещаем путь в очередь
                self.file_queue.put(file_path)
                # Выводим информацию о добавленном файле
                print(
                    f"[Producer] Добавлен файл: "
                    f"{file_path}"
                )
        print("[Producer] Сканирование завершено.")