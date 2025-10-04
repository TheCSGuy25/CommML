import numpy as np
from collections import Counter

class KNN:
    def __init__(self, n):
        self.k = n
        self.x = self.y = None

    def store(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def predict(self, x):
        x = np.array(x)
        self.distances = []
        len_x = len(self.x)
        for i in range(len_x):
            distance = np.sum((self.x[i] - x) ** 2)
            self.distances.append([distance, self.y[i]])

        self.distances.sort(key=lambda d: d[0])
        nearest_k = self.distances[:self.k]
        labels = [label for _, label in nearest_k]
        label_counts = Counter(labels)

        return label_counts.most_common(1)[0][0]
