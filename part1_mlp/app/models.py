import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        # Архитектура сети [.., .., ..]
        self.layer_sizes = layer_sizes
        # Cкорость обучения
        self.learning_rate = learning_rate
        # Матрицы весов
        self.weights = []
        # Смещения нейронов
        self.biases = []
        # Записи значения ошибок
        self.loss_history = []
        # Инициализация весов и смещений
        self._initialize_weights()

    # Метод для инициализации весов и смещений
    def _initialize_weights(self):
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
        # Возвращает 0 для отрицательных значений или само значение для положительных.
        return np.maximum(0, x)
    @staticmethod
    # Метод для сравнения предсказания сети с правильным ответом и отображения, насколько сеть ошиблась
    def cross_entropy_loss(y_true, y_pred):
        # Добавляем маленькое число, чтобы избежать log(0).
        epsilon = 1e-12

        # Ограничиваем вероятности безопасным диапазоном.
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Вычисляем cross-entropy для каждого объекта.
        loss = -np.sum(y_true * np.log(y_pred), axis=1)

        # Возвращаем среднюю ошибку по всему batch.
        return np.mean(loss)
    @staticmethod
    # Метод для превращения выходных значений нейросети в вероятности классов
    def softmax(x):
        # Вычитаем максимальное значение для численной стабильности.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        # Преобразуем значения в вероятности.
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    # Метод для проведения входных данных через всю нейросеть вперёд и получения предсказания.
    def forward(self, x):
        # Сохраняем активации всех слоёв
        activations = [x]
        # Начинаем с входных данных
        activation = x

        # Обрабатываем скрытые слои
        for i in range(len(self.weights) - 1):
            # Линейная комбинация входов, весов и смещений
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            # Применяем функцию активации ReLU
            activation = self.relu(z)

            # Сохраняем результат слоя для backpropagation
            activations.append(activation)

        # Вычисляем значение последнего слоя
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        # Применяем Softmax для получения вероятностей
        output = self.softmax(z)
        # Сохраняем выходной слой
        activations.append(output)

        # Возвращаем и результат, и промежуточные значения
        return output, activations
    # Метод, который обучает нейросеть, изменяя её веса на основе ошибки
    def backward(self, x, y_true, activations):
        # Количество объектов в текущем batch
        batch_size = x.shape[0]

        # Ошибка на текущем слое, которую нужно передать назад
        layer_error = activations[-1] - y_true

        # Проходим по слоям в обратном направлении
        for i in reversed(range(len(self.weights))):
            # Получаем активацию предыдущего слоя
            previous_activation = activations[i]
            # Вычисляем градиент весов.
            weight_gradient = np.dot(previous_activation.T, layer_error) / batch_size
            # Вычисляем градиент смещений.
            bias_gradient = np.sum(layer_error, axis=0, keepdims=True) / batch_size

            # Передаём ошибку на предыдущий слой
            if i > 0:
                # Распространяем ошибку назад через веса
                previous_layer_error = np.dot(layer_error, self.weights[i].T)
                # Применяем производную ReLU.
                layer_error = previous_layer_error * (activations[i] > 0)
            # Обновляем веса
            self.weights[i] -= self.learning_rate * weight_gradient
            # Обновляем смещения
            self.biases[i] -= self.learning_rate * bias_gradient
    # Метод для обучения нейросети на предоставленных данных
    def train(self, x, y_true, epochs=1000):
        # Очищаем историю ошибки перед новым обучением
        self.loss_history = []

        # Повторяем обучение указанное количество эпох
        for epoch in range(epochs):
            # Выполняем прямое распространение данных через сеть
            predictions, activations = self.forward(x)
            # Вычисляем ошибку текущего предсказания.
            loss = self.cross_entropy_loss(y_true, predictions)
            # Сохраняем значение ошибки для последующего анализа
            self.loss_history.append(loss)
            # Выполняем обратное распространение ошибки и обновляем веса и смещения
            self.backward(x, y_true, activations)
        # Возвращаем историю ошибки после завершения обучения
        return self.loss_history