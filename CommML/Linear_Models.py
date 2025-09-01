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
