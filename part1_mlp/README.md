# Part 1 — MLP Neural Network

Простая реализация многослойного перцептрона (MLP) на Python и NumPy.

Проект реализует базовый цикл работы нейронной сети:

- создание архитектуры сети;
- инициализация весов;
- прямое распространение (`forward`);
- функции активации;
- вычисление ошибки;
- обратное распространение (`backward`);
- обучение;
- предсказание;
- оценку точности;
- сохранение и загрузку весов;
- подготовку датасета.

## Структура проекта

```text
part1_mlp/
│
├── README.md
│
├── app/
│   ├── exceptions.py
│   ├── main.py
│   ├── manager.py
│   ├── models.py
│   └── utils.py
│
├── data/
│
└── weights/


Требования: 

- Python 3.10+
- NumPy
- Pandas


Запуск: 

1. Установить зависимости:
# Устанавливаем NumPy и Pandas.
python -m pip install numpy pandas 

2. Запуск проекта с параметрами по умолчанию:
# Запускаем основную программу.
python -m part1_mlp.app.main

3. Параметры обучения
# Запускаем обучение с 1000 эпохами и скоростью обучения 0.05.
python -m part1_mlp.app.main --epochs 1000 --learning-rate 0.05

4. Справка программыа
python -m part1_mlp.app.main --help

5. Сохранение модели -> weights/xor_model.npy

- Пример результата: 

Loss before: 5.237831397435584
Loss after: 0.036636559236337636
Weights saved
Predictions: [0 1 1 0]
Accuracy: 100.00%

- При повторном запуске 

Weights loaded
Predictions: [0 1 1 0]
Accuracy: 100.00%


Основные методы: 

forward() - Выполняет прямое распространение данных через сеть.

backward() - Вычисляет градиенты для обновления весов.

train() - Обучает нейронную сеть.

predict() - Возвращает предсказанные классы.

evaluate() - Вычисляет точность модели.

save_weights() - Сохраняет обученные веса.

load_weights() - Загружает ранее сохранённые веса.

Обработка ошибок

В проекте используются собственные исключения (app/exceptions.py):

InvalidLayerSizeError
MismatchedDataError
ModelNotInitializedError
DatasetError
DatasetFileNotFoundError
InvalidInputError
WeightFileError