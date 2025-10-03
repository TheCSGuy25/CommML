import math

class linear_regression:
  def __init__(self):
    self.slope = []
    self.intercept = 0
    self.learning_rate = 0.001

  def fit(self, x, y, epochs):
    data_size = len(x)
    number_of_features = len(x[0])
    
    self.slope = [0.0] * number_of_features
    self.intercept = 0.0

    for epoch in range(epochs):
        pred = []
        for i in range(data_size):
            y_hat = sum([_x * _slope for _x, _slope in zip(x[i], self.slope)]) + self.intercept
            pred.append(y_hat)

        intercept_gradient = (2 / data_size) * sum([(pred[j] - y[j]) for j in range(data_size)])
        
        slope_gradients = [0.0] * number_of_features
        for feature_idx in range(number_of_features):
            slope_gradients[feature_idx] = (2 / data_size) * sum(
                (pred[j] - y[j]) * x[j][feature_idx] for j in range(data_size)
            )

        self.intercept -= self.learning_rate * intercept_gradient
        for feature_idx in range(number_of_features):
            self.slope[feature_idx] -= self.learning_rate * slope_gradients[feature_idx]

        error = (1 / data_size) * sum([(pred[j] - y[j]) ** 2 for j in range(data_size)])
        print(f"Epoch {epoch+1}, MSE: {error:.4f}")



  def predict(self, x):
    return sum([_x * _slope for _x, _slope in zip(x, self.slope)]) + self.intercept




  
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


