import numpy as np
from tqdm import tqdm

class linear_regression:
    def __init__(self):
        self.weights = []
        self.bias = 0.0
        self.learning_rate = 0.001

    def fit(self, x, y, epochs=10):
        data_size = len(x)
        x = np.array(x)
        y = np.array(y)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        data_size, number_of_features = x.shape
        
        self.weights = np.zeros(number_of_features)

        for epoch in range(epochs):
            derivatives = [0.0] * number_of_features
            bias_derivative = 0.0

            with tqdm(total=data_size, desc=f"Epoch {epoch+1}/{epochs}", unit="sample") as pbar:
                for pos in range(data_size):

                    print(f"weights: {self.weights}, x[pos]: {x[pos]}")
                    prediction = sum([self.weights[i] * x[pos][i] for i in range(number_of_features)]) + self.bias

                    for i in range(number_of_features):
                        derivatives[i] += (2 / data_size) * (prediction - y[pos]) * x[pos][i]

                    bias_derivative += (2 / data_size) * (prediction - y[pos])
                    pbar.update(1)

            for i in range(number_of_features):
                self.weights[i] -= self.learning_rate * derivatives[i]

            self.bias -= self.learning_rate * bias_derivative
        

            if any([np.isnan(w) or np.isinf(w) or abs(w) > 1e10 for w in self.weights]):
                return

    def predict(self, x): 
        return sum([self.weights[i] * x[i] for i in range(len(self.weights))]) + self.bias if type(x) == list else self.weights[0] * x + self.bias



class polynomial_regression:
    def __init__(self):
        self.weights = []
        self.bias = 0.0
        self.learning_rate = 1e-7  
        self.history = {"loss": []}
    def fit(self, x, y, epochs=10):
        x = np.array(x)
        y = np.array(y)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        data_size, number_of_features = x.shape
        self.weights = [0.0] * number_of_features

        for epoch in range(epochs):
            with tqdm(total=data_size, desc=f"Epoch {epoch+1}/{epochs}", unit="sample") as pbar:
                for pos in range(data_size):
                    prediction = sum([self.weights[i] * (x[pos][i] ** (i + 1)) for i in range(number_of_features)]) + self.bias
                    for i in range(number_of_features):
                        self.weights[i] += (-2/data_size) * (x[pos][i] ** (i + 1)) * (y[pos] - prediction)
                    self.bias += (-2/data_size) * (y[pos] - prediction)
                    pbar.update(1)

            for i in range(number_of_features):
                self.weights[i] -= self.learning_rate * (self.weights[i] / data_size)

            self.bias -= self.learning_rate * (self.bias / data_size)

            if any(np.isnan(w) or np.isinf(w) or w > 1e10 for w in self.weights):
                return

    def predict(self, x):
        return sum([self.weights[i] * (x[i] ** (i + 1)) for i in range(len(self.weights))]) + self.bias if type(x) == list else self.weights[0] * (x) + self.bias



class logistic_regression:
    def __init__(self):
        self.theta = None
        self.LR = 0.001
    def sigmoid(self, z_input):
        z_input = np.clip(z_input, -300, 300)
        return 1 / (1 + np.exp(-z_input))

    def fit(self, x, y, epochs=10):
        x = np.array(x)
        y = np.array(y)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        data_size, number_of_features = x.shape
        self.theta = np.zeros(number_of_features)

        for epoch in range(epochs):
            tp = tn = fp = fn = 0
            with tqdm(total=data_size, desc=f"Epoch {epoch+1}/{epochs}", unit="sample") as pbar:
                for i in range(data_size):
                    z = self.sigmoid(np.dot(self.theta, x[i]))
                    gradient = (z - y[i]) * x[i]
                    self.theta -= self.LR * gradient

                    pred = 1 if z >= 0.5 else 0
                    if pred == 1 and y[i] == 1:
                        tp += 1
                    elif pred == 1 and y[i] == 0:
                        fp += 1
                    elif pred == 0 and y[i] == 1:
                        fn += 1
                    elif pred == 0 and y[i] == 0:
                        tn += 1
                    pbar.update(1)

            if any(np.isnan(w) or np.isinf(w) or w > 1e10 for w in self.theta):
                return
    def predict(self, x):
        x = np.array(x, dtype=np.float64)
        probs = self.sigmoid(np.dot(x, self.theta))
        return 1 if probs >= 0.5 else 0
