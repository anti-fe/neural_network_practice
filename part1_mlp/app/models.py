import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []

        self.loss_history = []

        self._initialize_weights()
    # Инициализация весов
    def _initialize_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
            biases = np.zeros((1, output_size))

            self.weights.append(weights)
            self.biases.append(biases)