import numpy as np
import tqdm
from CommML.Metrics import *
from CommML.Linear_Models import *
from CommML.misc import *

def regression_metrics(y_true, y_pred):
    metrics = {}
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    metrics["Mean Squared Error"] = mse
    metrics["Root Mean Squared Error"] = rmse
    metrics["Mean Absolute Error"] = mae
    return metrics
def regression_workflow(x, y, epochs=10):
    """
    Implementation of a regression workflow
    Input: X, Y, Epochs
    Output: Regression Model, Regression Metrics
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    model = linear_regression()
    model.fit(x_train, y_train, epochs=epochs)
    y_pred = model.predict(x_test)
    metrics = regression_metrics(y_test, y_pred)
    return model, metrics


def classification_metrics(y_true, y_pred):
    metrics = {}
    accuracy = get_accuracy(y_true, y_pred)
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics["Accuracy"] = accuracy
    metrics["Precision"] = precision
    metrics["Recall"] = recall
    metrics["F1 Score"] = f1
    return metrics

def classification_workflow(x, y, epochs=10):
    """
    Implementation of a classification workflow
    Input: X, Y, Epochs
    Output: Classification Model, Classification Metrics
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    model = logistic_regression()
    model.fit(x_train, y_train, epochs=epochs)
    y_pred = model.predict(x_test)
    metrics = classification_metrics(y_test, y_pred)
    return model, metrics