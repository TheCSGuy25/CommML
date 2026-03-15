import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# Load the official sklearn diabetes dataset (442 samples, 10 features)
diabetes = load_diabetes()
X_train = diabetes.data[:400]  # First 400 for training
y_train = diabetes.target[:400]
X_test = diabetes.data[400:]   # Last 42 for testing  
y_test = diabetes.target[400:]

print("Dataset loaded!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Features: {diabetes.feature_names}")
print(f"\nSample (first 3 rows):")
print("BMI    BP   S1   S2   S3   S4   S5   S6    GLU → Disease Progression")
# for i in range(3):
#     print(f"{X_train[i, 2]:5.1f} {X_train[i, 3]:3.1f} {X_train[i, 4:8]:.1f} {X_train[i, 9]:5.1f} → {y_train[i]:5.1f}")

# print("\nLinear Regression: \n")
lr = SklearnLR()
lr.fit(X_train, y_train)
lr.predict(X_test)
mean_squared_error_val = mean_squared_error(y_test, lr.predict(X_test))
mean_absolute_error_val = mean_absolute_error(y_test, lr.predict(X_test))
root_mean_squared_error_val = root_mean_squared_error(y_test, lr.predict(X_test))
print(f"Mean Squared Error: {mean_squared_error_val}\n")
print(f"Mean Absolute Error: {mean_absolute_error_val}\n")
print(f"Root Mean Squared Error: {root_mean_squared_error_val}\n")

print("\nCustom Linear Regression: \n")
from linear_models import linear_regression

lr = linear_regression()
lr.fit(X_train, y_train)
res = lr.predict(X_test)
mean_squared_error_val = mean_squared_error(y_test, res)
mean_absolute_error_val = mean_absolute_error(y_test, res)
root_mean_squared_error_val = root_mean_squared_error(y_test, res)
print(f"Mean Squared Error: {mean_squared_error_val}\n")
print(f"Mean Absolute Error: {mean_absolute_error_val}\n")
print(f"Root Mean Squared Error: {root_mean_squared_error_val}\n")