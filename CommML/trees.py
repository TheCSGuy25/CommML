import numpy as np


class DecsisionTree:
    def __init__(self):
        self.features = None
        self.x = self.y = None
        self.entropy = None

    def calculate_entropy(self,y):
        values , counts = np.unique(y,return_counts=True)
        length = len(y)
        self.entropy = -np.sum(counts[i]/length * np.log2(counts[i]/length) for i in range(len(values)))
        
    
    def information_gain(self, x, y):
        parent_entropy = self.calculate_entropy(y)

        values, counts = np.unique(x, return_counts=True)
        total = len(x)
        
        second_part = 0
        for v, count in zip(values, counts):
            y_v = y[x == v]
            second_part += (count / total) * self.calculate_entropy(y_v)

        information_gain = parent_entropy - second_part
        return information_gain


