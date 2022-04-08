import numpy as np


class Distance:

    thresholds = {
        'VGG-Face': {
            'cosine': 0.40,
            'euclidean': 0.60,
            'euclidean_l2': 0.86
        },
        'Facenet': {
            'cosine': 0.40,
            'euclidean': 10,
            'euclidean_l2': 0.80
        },
        'Facenet512': {
            'cosine': 0.30,
            'euclidean': 23.56,
            'euclidean_l2': 1.04
        },
        'ArcFace': {
            'cosine': 0.68,
            'euclidean': 4.15,
            'euclidean_l2': 1.13
        },
        'Dlib': {
            'cosine': 0.07,
            'euclidean': 0.6,
            'euclidean_l2': 0.4
        },
        'OpenFace': {
            'cosine': 0.10,
            'euclidean': 0.55,
            'euclidean_l2': 0.55
        },
        'DeepFace': {
            'cosine': 0.23,
            'euclidean': 64,
            'euclidean_l2': 0.64
        },
        'DeepID': {
            'cosine': 0.015,
            'euclidean': 45,
            'euclidean_l2': 0.17
        }
    }

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
