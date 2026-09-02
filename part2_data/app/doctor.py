import os


def check_environment():
    """Проверяет окружение проекта"""

    # Проверяем каталог с исходными данными
    data_path = os.getenv("DATA_PATH", "part2_data/data")
    # Проверяем каталог для результатов
    output_path = os.getenv("OUTPUT_PATH", "part2_data/output/data.npy")
    # Получаем каталог, в котором должен находиться итоговый файл
    output_dir = os.path.dirname(output_path)
    # Выводим информацию о текущем окружении
    print("Проверка окружения:")
    print(f"Каталог данных: {data_path}")
    print(f"Каталог результата: {output_dir}")
    
    # Проверяем наличие каталога с данными
    if os.path.isdir(data_path):
        print("OK: каталог данных существует.")
    else:
        print("ERROR: каталог данных не найден.")

    # Создаём каталог output, если его ещё нет
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print("OK: каталог output существует.")