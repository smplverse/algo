import numpy as np


class Distance:

    def __init__(self, source: np.ndarray, test: np.ndarray):
        self.source = source
        self.test = test

    def cosine(self) -> float:
        a = np.matmul(np.transpose(self.source), self.test)
        b = np.sum(np.multiply(self.source, self.source))
        c = np.sum(np.multiply(self.test, self.test))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def euclidean(self) -> float:
        distance = self.source - self.test
        distance = np.sum(np.multiply(distance, distance))
        distance = np.sqrt(distance)
        return distance

    def l2_normalize(x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))
