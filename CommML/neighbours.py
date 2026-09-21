import numpy as np
from collections import Counter
import math

class KNN:
    def __init__(self, k, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.x = self.y = None

    def fit(self, x, y):
        if self.k > len(x):
            raise ValueError("k must be less than or equal to the number of samples.")
        self.x = np.array(x)
        self.y = np.array(y)

    def __distance(self, x1, x2):
        if self.metric == 'euclidean':
            return math.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def predict(self, x):
        x = np.array(x)
        distances = []
        
        for i in range(len(self.x)):
            d = self.__distance(self.x[i], x)
            distances.append((d, self.y[i]))

        distances.sort(key=lambda d: d[0])
        nearest_k = distances[:self.k]
        labels = [label for _, label in nearest_k]
        label_counts = Counter(labels)
        pred = label_counts.most_common(1)[0][0]
        return pred