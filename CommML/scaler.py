import numpy as np
import random

def normalization(x):
    """
    A scaling method that scales the values of X based on the 
    minumum and maximum value
    
    :param x: A variable feature length input
    :return: A variable min output
    """

    number_of_features = len(x[0]) if type(x[0]) == list else 1
    x = np.array(x)

    if number_of_features == 1:
        minmum_x = min(x)
        maximum_x = max(x)
        return ((x - minmum_x) / (maximum_x - minmum_x)).tolist()
    else:
        min_x = np.min(x, axis=0)
        max_x = np.max(x, axis=0)

        if any(max_x[i] == min_x[i] for i in range(number_of_features)):
            raise ValueError("A minumum and maxium value is the same which leads to division by Zero")
        
        return ((x - min_x) / (max_x - min_x)).tolist()

def mean_scaler(x):
    number_of_features  = len(x[0]) if type(x[0]) == list else 1
    x = np.array(x)
    if number_of_features == 1:
        mean_x = np.mean(x)
        min_x, max_x = np.min(x), np.max(x)
        return ((x - mean_x) / (max_x - min_x)).tolist()
    else:
        mean_x = np.mean(x, axis=0)
        min_x = np.min(x, axis=0)
        max_x = np.max(x, axis=0)

        if np.any(max_x == min_x):
            raise ValueError("A feature has equal min and max; division by zero.")

        return ((x - mean_x) / (max_x - min_x)).tolist()
    
def standard_scaler(x):
    x = np.array(x)
    return ((x - np.mean(x)) / np.std(x)).tolist()


