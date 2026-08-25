class NeuralNetworkError(Exception):
    # Базовое исключение для ошибок нейронной сети
    pass

class InvalidLayerSizeError(NeuralNetworkError):
    # Возникает при указании некорректного размера слоя
    pass

class MismatchedDataError(NeuralNetworkError):
    # Возникает при несовместимых размерах входных и целевых данных
    pass

class ModelNotInitializedError(NeuralNetworkError):
    # Возникает при попытке использовать неинициализированную модель
    pass

class DatasetError(NeuralNetworkError):
    # Базовое исключение для ошибок работы с датасетом
    pass

class DatasetFileNotFoundError(DatasetError):
    # Возникает, если файл датасета не найден
    pass

class InvalidInputError(NeuralNetworkError):
    # Возникает при некорректном пользовательском вводе
    pass

class WeightFileError(NeuralNetworkError):
    # Возникает при ошибке сохранения или загрузки весов
    pass