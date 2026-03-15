import numpy as np

def check_x_and_y(x, y):
    if x.ndim < 2:
        raise ValueError("X must have at least 2 features.")
    if len(x) != len(y):
        raise ValueError("X and y must have the same length.")
    
    return True

