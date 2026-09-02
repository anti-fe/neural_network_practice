import argparse
import os
import numpy as np

from dotenv import load_dotenv
from .doctor import check_environment
from .data_processor import prepare_data

# Загружаем переменные окружения из /.env
load_dotenv()

def parse_args():
    """Создаёт и обрабатывает аргументы командной строки."""

    # Создаём главный парсер команд
    parser = argparse.ArgumentParser(
        description="Модуль подготовки данных для нейронной сети"
    )
    # Подкоманды prepare и doctor
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    # Команда prepare
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Подготовка файлов данных",
    )
    # Путь к каталогу с данными
    prepare_parser.add_argument(
        "--path",
        default=os.getenv(
            "DATA_PATH",
            "part2_data/data",
        ),
        help="Путь к каталогу с файлами данных",
    )
    # Расширения файлов для поиска
    prepare_parser.add_argument(
        "--ext",
        nargs="+",
        default=os.getenv(
            "DATA_EXT",
            ".csv",
        ).split(),
        help="Расширения файлов, например .csv .png .jpg",
    )
    # Путь к итоговому файлу
    prepare_parser.add_argument(
        "--output",
        default=os.getenv(
            "OUTPUT_PATH",
            "part2_data/output/data.npy",
        ),
        help="Путь к файлу .npy для сохранения результата",
    )
    # Команда doctor
    subparsers.add_parser("doctor",help="Проверка окружения")
    return parser.parse_args()


def main():
    """Основная точка входа программы."""

    # Аргументы командной строки
    args = parse_args()
    # Определяем выбранную команду
    if args.command == "prepare":
        print("Команда prepare выбрана")
        print(f"Путь: {args.path}")
        print(f"Расширения: {args.ext}")
        print(f"Выходной файл: {args.output}")
        try:
            # Загружаем, нормализуем и объединяем данные
            data, statistics = prepare_data(args.path,args.ext)
            # Создаём каталог для выходного файла
            output_dir = os.path.dirname(args.output)

            if output_dir:
                # Создаём каталог, если его ещё нет
                os.makedirs(output_dir, exist_ok=True)
            # Сохраняем итоговый массив
            np.save(args.output,data)

            # Выводим статистику обработки
            print("\nПодготовка завершена")
            print(f"Файлов обработано: {statistics['file_count']}")
            print(f"Размер итогового массива: {statistics['shape']}")
            print(f"Тип данных: {statistics['dtype']}")
            print(
                f"Время обработки: "
                f"{statistics['processing_time']:.4f} сек."
            )
            print("Исходные размеры:")
            # Выводим информацию по каждому файлу
            for item in statistics["original_shapes"]:
                print(
                    f"  {item['file']}: "
                    f"{item['shape']}"
                )
            print(f"Файл сохранён: {args.output}")

        except (FileNotFoundError, ValueError) as error:
            print(f"Ошибка: {error}")
    elif args.command == "doctor":
        # Запускаем проверку окружения
        check_environment()


if __name__ == "__main__":
    # Запускаем программу.
    main()