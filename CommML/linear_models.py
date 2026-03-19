import numpy as np
import scipy.linalg
from sklearn.preprocessing import PolynomialFeatures

from CommML.utils import check_x_and_y

class linear_regression:
    """Linear regression using SVD-based least squares solver."""
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.x_mean = None
        self.y_mean = None
        self.x_shape = None
        self.is_fitted = False

    def fit(self, x, y):
        """
        Fit linear regression model using SVD.
        
        Args:
            x:  2D array of shape (n_samples, n_features)
            y: 1D array of shape (n_samples,)
        """
        x = np.array(x , dtype=float)
        y = np.array(y , dtype=float)
        
        if check_x_and_y(x, y):
            self.x_shape = x.shape[1]
            self.x_mean = np.mean(x , axis=0)
            self.y_mean = np.mean(y)
            

            x_centered = x - self.x_mean
            y_centered = y - self.y_mean
            

            self.coefficients, _, self.matrix_rank, self.singular_values = scipy.linalg.lstsq(x_centered, y_centered)

            self.intercept = self.y_mean - np.dot(self.x_mean, self.coefficients)

            self.is_fitted = True
        else:
            raise ValueError("Feature matrix and target vector must have compatible shapes.")
        
        return self
    
    def predict(self, x):
        """
        Predict target values for new data.
        
        Args:
            x: 2D array of shape (n_samples, n_features)
            
        Returns:
            Predicted targets, shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Please run fit() on x and y first.")
        
        x = np.array(x , dtype=float)
        
        if x.shape[1] != self.x_shape:
            raise ValueError("Shape Mismatch of train and predict data. Train: {self.x_shape}, Predict: {x.shape[1]}")
        
        x_centered = x - self.x_mean
        
        predictions = np.dot(x_centered, self.coefficients) + self.intercept
        
        return predictions


class polynomial_regression:
    """Polynomial regression using PolynomialFeatures + linear_regression."""

    def __init__(self, degree):
        self.degree = degree
        self.model = linear_regression()
        self.polynomial = None
        self.is_fitted = False

    def fit(self, x, y):
        """
        Fit polynomial regression model.
        Args:
            x: 2D array of shape (n_samples, n_features)
            y: 1D array of shape (n_samples,)
        """
        self.polynomial = PolynomialFeatures(degree=self.degree, include_bias=False)
        x_poly = self.polynomial.fit_transform(x)
        self.model.fit(x_poly, y)
        self.is_fitted = True
        return self

    def predict(self, x):
        """
        Predict target values for new data.
        Args:
            x: 2D array of shape (n_samples, n_features)
        Returns:
            Predicted targets, shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Please run fit() on x and y first.")
        x_poly = self.polynomial.transform(x)
        return self.model.predict(x_poly)
