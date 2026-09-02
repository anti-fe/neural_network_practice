import os


def find_data_files(data_path, extensions):
    """Находит файлы данных с указанными расширениями"""

    # Список найденных файлов
    data_files = []
    # Проверка существования каталога
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Каталог с данными не найден: {data_path}"
        )

    # Обход каталога и всех вложенных каталогов
    for root, _, files in os.walk(data_path):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()
            # Сравниваем расширение с разрешёнными
            if extension in extensions:
                # Добавляем полный путь к найденному файлу
                data_files.append(
                    os.path.join(root, filename)
                )

    # Возвращаем список найденных файлов
    return data_files