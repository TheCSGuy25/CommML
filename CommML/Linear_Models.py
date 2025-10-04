import numpy as np

class linear_regression:
    def __init__(self):
        self.weights = []
        self.bias = 0.0
        self.learning_rate = 0.001

    def fit(self, x, y, epochs):
        data_size = len(x)
        number_of_features = len(x[0])
        x = np.array(x)
        y = np.array(y)
        self.weights = np.zeros(number_of_features)

        for epoch in range(epochs):
            derivatives = [0.0] * number_of_features
            bias_derivative = 0.0

            for pos in range(data_size):
                prediction = sum([self.weights[i] * x[pos][i] for i in range(number_of_features)]) + self.bias

                for i in range(number_of_features):
                    derivatives[i] += (2 / data_size) * (prediction - y[pos]) * x[pos][i]

                bias_derivative += (2 / data_size) * (prediction - y[pos])

            print(f"Epoch {epoch+1} | Weights: {self.weights} | Bias: {self.bias}")

            for i in range(number_of_features):
                self.weights[i] -= self.learning_rate * derivatives[i]

            self.bias -= self.learning_rate * bias_derivative

            if any([np.isnan(w) or np.isinf(w) or abs(w) > 1e10 for w in self.weights]):
                print("Weights exploding or invalid, terminating training.")
                return

    def predict(self, x):
        return sum([self.weights[i] * x[i] for i in range(len(self.weights))]) + self.bias


  
class polynomial_regression:
  def __init__(self):
    self.weights = []
    self.bias = 0.0
    self.learning_rate = 1e-7  


  def fit(self, x, y, epochs):
    x_len = len(x)
    number_of_features = len(x[0])
    x = np.array(x)
    y = np.array(y)
    self.weights = np.zeros(number_of_features)

    for epoch in range(epochs):
      derivatives = [0.0] * number_of_features
      bias_derivative = 0.0

      for pos in range(x_len):
        prediction = sum([self.weights[i] * (x[pos][i] ** (i + 1)) for i in range(number_of_features)]) + self.bias

        for i in range(number_of_features):
          derivatives[i] += (-2/x_len) * (x[pos][i] ** (i + 1)) * (y[pos] - prediction)

        bias_derivative += (-2/x_len) * (y[pos] - prediction)

      print(f"Epochs: {epoch} | Weights: {self.weights}")
      for i in range(number_of_features):
        self.weights[i] -= self.learning_rate * (derivatives[i] / x_len)

      self.bias -= self.learning_rate * (bias_derivative / x_len)
      if any(np.isnan(w) or np.isinf(w) or w > 1e10 for w in self.weights):
                print("Weights exploding or invalid, terminating training.")
                return


  def predict(self, x):
    return sum([self.weights[i] * (x[0][i] ** (i + 1)) for i in range(len(self.weights))]) + self.bias



class LogisticRegression:
    def __init__(self):
        self.theta = None
        self.LR = 0.001

    def sigmoid(self, z_input):
        z_input = np.clip(z_input, -300, 300)  # to prevent overflow as largest supported value: -1.79e+308 | smallest: 2.22e-308 
        return 1 / (1 + np.exp(-z_input))

    def fit(self, x, y, epochs):
        x = np.array(x)
        y = np.array(y)
        self.theta = np.zeros(x.shape[1])

        for epoch in range(epochs):
            tp = tn = fp = fn = 0
            for i in range(len(x)):
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

            acc = (tp + tn) / (tp + tn + fp + fn)
            print(f"Epoch {epoch + 1}/{epochs} ........................................ Accuracy: {acc:.4f}")

    def predict_prob(self, x):
        x = np.array(x, dtype=np.float64)
        return self.sigmoid(np.dot(x, self.theta))

    def predict(self, x):
        probs = self.predict_prob(x)
        return [1 if p >= 0.5 else 0 for p in probs]


