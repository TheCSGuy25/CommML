import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, metric='euclidean', algorithm='brute'):
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.x = self.y = None
        self.is_trained = False

    def fit(self, x, y):
        if self.k > len(x):
            raise ValueError("k must be less than or equal to the number of samples.")
        self.x = np.array(x)
        self.y = np.array(y)

    def __distance(self, x1, x2):
        """Compute distance between two points based on self.metric."""
        if self.metric == 'euclidean':
            return np.sum((x1 - x2) ** 2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def predict(self, x):
        x = np.array(x)
        if not self.is_trained:
            raise ValueError("Model has not been trained. Please run fit() on x and y first.")
        
        if self.algorithm != 'brute':
            raise ValueError("Only 'brute' algorithm is implemented.")
        
        predictions = []
        for xt in x:
            distances = []
            for i in range(len(self.x)):
                d = self.__distance(self.x[i], xt)
                distances.append((d, self.y[i]))

            distances.sort(key=lambda d: d[0])
            nearest_k = distances[:self.k]
            labels = [label for _, label in nearest_k]
            label_counts = Counter(labels)
            pred = label_counts.most_common(1)[0][0]
            predictions.append(pred)

        return np.array(predictions)