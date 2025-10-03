import math

class linear_regression:
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.learning_rate = 0.001

    def fit(self, x, y, epochs):
        data_size = len(x)
        number_of_features = len(x[0])
        self.weights = [0.0] * number_of_features
        self.bias = 0.0

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

            if any([math.isnan(w) or math.isinf(w) or abs(w) > 1e10 for w in self.weights]):
                print("Weights exploding or invalid, terminating training.")
                return

    def predict(self, x):
        return sum([self.weights[i] * x[i] for i in range(len(self.weights))]) + self.bias


  
class polynomial_regression:
  def __init__(self):
    self.weights = []
    self.bias = 0
    self.learning_rate = 1e-7  


  def fit(self, x, y, epochs):
    x_len = len(x)
    number_of_features = len(x[0])
    self.weights = [0.0] * number_of_features

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
      if any(math.isnan(w) or math.isinf(w) or w >1e10 for w in self.weights):
        print("Weights exploding or invalid, terminating training.")
        return


  def predict(self, x):
    return sum([self.weights[i] * (x[0][i] ** (i + 1)) for i in range(len(self.weights))]) + self.bias


