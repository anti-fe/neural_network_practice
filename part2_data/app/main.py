import argparse
import os


def parse_args():
    """Создаёт и обрабатывает аргументы командной строки."""

    # Создаём главный парсер команд.
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
    subparsers.add_parser(
        "doctor",
        help="Проверка окружения",
    )
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

    elif args.command == "doctor":
        print("Команда doctor выбрана")


if __name__ == "__main__":
    # Запускаем программу.
    main()