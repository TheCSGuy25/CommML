import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        U , S, V_T = np.linalg.svd(X, full_matrices=False)
        s_inverse = np.array([1/s if s >= 1e-10 else 0 for s in S])
        self.weights = V_T.T @ np.diag(s_inverse) @ U.T @ Y

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights