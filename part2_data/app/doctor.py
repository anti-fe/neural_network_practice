import os
import platform
import subprocess
import sys


def check_python():
    """Проверяет версию Python."""
    # Получаем текущую версию Python
    version = sys.version.split()[0]
    # Выводим версию Python
    print(f"Python: {version}")

def check_library(module_name, package_name=None):
    """Проверяет наличие Python-библиотеки."""
    # Если имя пакета не передано, используем имя модуля
    if package_name is None:
        package_name = module_name
    try:
        # Пытаемся импортировать библиотеку
        module = __import__(module_name)
        # Получаем версию установленной библиотеки
        version = getattr(module, "__version__", "версия неизвестна")
        print(f"{package_name}: OK ({version})")

    except ImportError:
        print(f"{package_name}: НЕ УСТАНОВЛЕНА")

def check_cuda():
    """Проверяет доступность NVIDIA CUDA через nvidia-smi."""
    print("CUDA / NVIDIA:")
    try:
        # Запускаем внешнюю команду nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Проверяем код завершения команды
        if result.returncode == 0:
            print("nvidia-smi: OK")
            # Выводим первые строки результата
            lines = result.stdout.splitlines()
            for line in lines[:5]:
                print(f"  {line}")

        else:
            print("nvidia-smi: недоступен")

    except FileNotFoundError:
        print("nvidia-smi: команда не найдена")
    except subprocess.TimeoutExpired:
        print("nvidia-smi: превышено время ожидания")

def check_working_directory():
    """Показывает текущую рабочую директорию."""
    # Получаем текущую рабочую директорию
    current_directory = os.getcwd()
    # Выводим путь
    print(f"Рабочая директория: {current_directory}")

def check_environment():
    """Выполняет полную проверку окружения."""
    # Выводим заголовок диагностики
    print("=== Проверка окружения ===")
    # Проверяем версию Python
    check_python()
    # Проверяем NumPy
    check_library("numpy", "NumPy")
    # Проверяем Pillow
    check_library("PIL", "Pillow")
    # Проверяем PyTorch
    check_library("torch", "PyTorch")
    # Проверяем TensorFlow
    check_library("tensorflow", "TensorFlow")
    # Проверяем наличие CUDA через nvidia-smi
    check_cuda()
    # Показываем текущую рабочую директорию
    check_working_directory()
    # Выводим информацию об операционной системе
    print(f"ОС: {platform.system()} {platform.release()}")
    # Завершаем диагностику
    print("=== Проверка завершена ===")