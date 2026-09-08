import numpy as np
import os

from part1_mlp.app.exceptions import (
    InvalidLayerSizeError,
    ModelNotInitializedError,
    WeightFileError,
)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, activation="relu"):
        # Проверяем, что архитектура сети задана в виде списка или кортежа
        if not isinstance(layer_sizes, (list, tuple)):
            raise InvalidLayerSizeError(
                "layer_sizes должен быть list или tuple"
            )
        # Проверяем, что указано минимум два слоя: входной и выходной
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError(
                "Neural network должен содержать как минимум два слоя"
            )
        # Проверяем, что размер каждого слоя является положительным числом.
        if any(
            not isinstance(size, int) or size <= 0
            for size in layer_sizes
        ):
            raise InvalidLayerSizeError(
                "Layer sizes должны быть положительными целыми числами"
            )
        # Проверяем допустимую функцию активации.
        if activation not in ("relu", "sigmoid"):
            raise ValueError(
                "Функция активации должна быть relu или sigmoid"
            )
        # Архитектура сети [.., .., ..]
        self.layer_sizes = layer_sizes
        # Cкорость обучения
        self.learning_rate = learning_rate
        # Выбранная функция активации
        self.activation = activation
        # Матрицы весов
        self.weights = []
        # Смещения нейронов
        self.biases = []
        # Записи значения ошибок
        self.loss_history = []
        # Инициализация весов и смещений
        self._initialize_weights()

    def _initialize_weights(self):
        """Метод для инициализации весов и смещений"""
        # Инициализация весов и смещений для каждого слоя сети
        for i in range(len(self.layer_sizes) - 1):
            # Количество нейронов в текущем слое
            input_size = self.layer_sizes[i]
            # Количество нейронов в следующем слое
            output_size = self.layer_sizes[i + 1]
            # Инициализация весов с использованием нормального распределения
            weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
            # Инициализация смещений нулями
            biases = np.zeros((1, output_size))
            # Добавление весов и смещений в соответствующие списки 
            self.weights.append(weights)
            self.biases.append(biases)
    @staticmethod
    def relu(x):
        """Возвращает 0 для отрицательных значений или само значение для положительных."""
        return np.maximum(0, x)
    def sigmoid(self, x):
        """Метод для вычисления сигмоидальной функции активации"""
        # Вычисляем сигмоидальную функцию активации.
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """Метод для сравнения предсказания сети с правильным ответом и отображения, насколько сеть ошиблась"""
        # Добавляем маленькое число, чтобы избежать log(0).
        epsilon = 1e-12
        # Ограничиваем вероятности безопасным диапазоном.
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Вычисляем cross-entropy для каждого объекта.
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        # Возвращаем среднюю ошибку по всему batch.
        return np.mean(loss)
    @staticmethod
    def softmax(x):
        """Метод для превращения выходных значений нейросети в вероятности классов"""
        # Вычитаем максимальное значение для численной стабильности.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        # Преобразуем значения в вероятности.
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    def forward(self, x):
        """Метод для проведения входных данных через всю нейросеть вперёд и получения предсказания"""
        # Сохраняем активации всех слоёв
        activations = [x]
        # Начинаем с входных данных
        activation = x

        # Обрабатываем скрытые слои
        for i in range(len(self.weights) - 1):
            # Линейная комбинация входов, весов и смещений
            z = np.dot(activation, self.weights[i]) + self.biases[i]
             # Выбираем функцию активации скрытого слоя.
            if self.activation == "relu":
                activation = self.relu(z)
            else:
                activation = self.sigmoid(z)
            # Сохраняем активацию текущего слоя
            activations.append(activation)
        # Вычисляем значение последнего слоя
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        # Применяем Softmax для получения вероятностей
        output = self.softmax(z)
        # Сохраняем выходной слой
        activations.append(output)

        # Возвращаем и результат, и промежуточные значения
        return output, activations
    def backward(self, x, y_true, activations):
        """Выполняет обратное распространение ошибки и обновляет веса."""

        # Количество объектов в текущем batch
        batch_size = x.shape[0]
        # Ошибка выходного слоя
        layer_error = activations[-1] - y_true
        # Проходим по слоям в обратном направлении
        for i in reversed(range(len(self.weights))):
            # Получаем активацию предыдущего слоя
            previous_activation = activations[i]
            # Вычисляем градиент весов
            weight_gradient = (
                np.dot(previous_activation.T, layer_error)
                / batch_size
            )
            # Вычисляем градиент смещений
            bias_gradient = (
                np.sum(layer_error, axis=0, keepdims=True)
                / batch_size
            )
            # Если это не первый слой, передаём ошибку на предыдущий слой
            if i > 0:
                # Распространяем ошибку через веса
                previous_layer_error = np.dot(
                    layer_error,
                    self.weights[i].T
                )
                # Получаем активацию скрытого слоя.
                hidden_activation = activations[i]
                # Выбираем производную функции активации
                if self.activation == "relu":
                    activation_gradient = (hidden_activation > 0)
                else:
                    activation_gradient = (hidden_activation * (1 - hidden_activation))
                # Получаем ошибку предыдущего слоя
                layer_error = (
                    previous_layer_error
                    * activation_gradient
                )
            # Обновляем веса
            self.weights[i] -= (self.learning_rate * weight_gradient)
            # Обновляем смещения
            self.biases[i] -= (self.learning_rate * bias_gradient)          
    def train(self, x, y_true, epochs=1000, batch_size=32):
        """Метод для обучения нейросети на предоставленных данных"""
        # Очищаем историю ошибки перед новым обучением
        self.loss_history = []
        # Получаем количество объектов в датасете
        data_size = x.shape[0]
        # Запускаем обучение на заданное количество эпох
        for epoch in range(epochs):
            # Перемешиваем индексы объектов перед каждой эпохой
            indices = np.random.permutation(data_size)
            # Разбиваем датасет на небольшие batch
            for start in range(0, data_size, batch_size):
                # Определяем конец текущего batch
                end = start + batch_size
                # Получаем индексы текущего batch
                batch_indices = indices[start:end]
                # Получаем входные данные текущего batch
                x_batch = x[batch_indices]
                # Получаем правильные ответы текущего batch
                y_batch = y_true[batch_indices]
                # Выполняем прямое распространение
                predictions, activations = self.forward(x_batch)
                # Вычисляем ошибку текущего batch
                loss = self.cross_entropy_loss(y_batch, predictions)
                # Выполняем обратное распространение и обновляем веса
                self.backward(x_batch, y_batch, activations)
            # После завершения эпохи сохраняем её итоговую ошибку
            self.loss_history.append(loss)
        # Возвращаем историю обучения
        return self.loss_history
    def predict(self, x):
        """Метод для получения результата от уже обученной нейросети"""
        # Проверяем, что модель содержит веса.
        if not self.weights or not self.biases:
            raise ModelNotInitializedError(
                "Модель не была инициализирована весами"
            )
        # Передаём входные данные через нейросеть
        predictions, _ = self.forward(x)
        # Для каждого объекта выбираем класс с максимальной вероятностью
        predicted_classes = np.argmax(predictions, axis=1)
        # Возвращаем номера предсказанных классов
        return predicted_classes
    def save_weights(self, file_path):
        """Метод для сохранения весов и смещений нейросети в файл"""
        np.save(
            file_path,
            {
                "weights": self.weights,
                "biases": self.biases,
            },
            allow_pickle=True,
        )
    def load_weights(self, file_path):
        """Метод для загрузки весов и смещений нейросети из файла"""
        if not os.path.isfile(file_path):
            raise WeightFileError(
                f"Weight file не найден: {file_path}"
            )

        try:
            data = np.load(
                file_path,
                allow_pickle=True,
            ).item()
            # Проверяем наличие весов
            if "weights" not in data:
                raise WeightFileError(
                    "Weight file не содержит weights"
                )
            # Проверяем наличие смещений
            if "biases" not in data:
                raise WeightFileError(
                    "Weight file не содержит biases"
                )
            # Восстанавливаем веса нейронной сети
            self.weights = data["weights"]
            # Восстанавливаем смещения нейронной сети
            self.biases = data["biases"]
        except WeightFileError:
            raise
        except Exception as error:
            raise WeightFileError(
                f"Ошибка загрузки weights: {error}"
            ) from error
    def evaluate(self, x, y_true):
        """Метод для оценки точности (в процентах) нейросети на тестовых данных"""
        # Получаем предсказанные классы для тестовых данных.
        predictions = self.predict(x)

        # Если y_true представлен в one-hot формате, преобразуем его обратно в номера классов
        if y_true.ndim > 1:
            true_classes = np.argmax(y_true, axis=1)
        else:
            true_classes = y_true

        # Считаем количество правильных предсказаний
        correct = np.sum(predictions == true_classes)
        # Вычисляем общее количество объектов
        total = len(true_classes)
        # Вычисляем точность модели
        accuracy = correct / total
        # Возвращаем accuracy
        return accuracy