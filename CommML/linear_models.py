import numpy as np
from itertools import combinations_with_replacement

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

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.weights = None

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        columns = [np.ones(n_samples)]
        for degree in range(1, self.degree + 1):
            for feature_indices in combinations_with_replacement(range(n_features), degree):
                columns.append(np.prod(X[:, feature_indices], axis=1))

        return np.column_stack(columns)

    
    def fit(self, X, y):
        y = np.array(y)
        X_poly = self._transform(X)
        U, S, Vt = np.linalg.svd(X_poly, full_matrices=False)
        s_inverse = np.array([ 1 / value if value >= 1e-10 else 0 for value in S ])
        self.weights = Vt.T @ np.diag(s_inverse) @ U.T  @ y

        return self

    def predict(self, X):
        return self._transform(X) @ self.weights

