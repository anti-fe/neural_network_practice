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
        activation = x

        # Проходим по всем слоям кроме последнего.
        for i in range(len(self.weights) - 1):
            # Вычисляем линейную комбинацию: Z = X * W + B.
            z = np.dot(activation, self.weights[i]) + self.biases[i]

            # Применяем ReLU к результату.
            activation = self.relu(z)

        # Обрабатываем последний слой отдельно.
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]

        # Используем Softmax, чтобы получить вероятности классов.
        output = self.softmax(z)

        return output