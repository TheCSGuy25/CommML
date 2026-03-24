import numpy as np

class normalization:
    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.x_dim = None

    def __check_fit(self):
        if self.x_max is None and self.x_min is None:
            raise ValueError("Model has not been trained. Please run fit() on x first.")

    def transform(self, x):
        x = np.array(x)
        self.__check_fit()
        return (x - self.x_min) / (self.x_max - self.x_min)
    
    def __fit(self, x):
        x = np.array(x)
        if x.ndim == 1:
            self.x_min = np.min(x)
            self.x_max = np.max(x)
        elif x.ndim > 1:
            self.x_min = np.min(x, axis=0)
            self.x_max = np.max(x, axis=0)

    def fit_transform(self, x):
        self.__fit(x)
        return self.transform(x)
        
