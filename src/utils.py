import numpy as np


def preprocess(data):
    """Простая нормализация данных"""
    data = np.array(data)
    return (data - np.mean(data)) / np.std(data)