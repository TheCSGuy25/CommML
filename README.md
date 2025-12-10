# CommML
A community developed ML Library

## Installation
To install the CommML library from pip:
```py
pip install CommML
```

## Example Usage
```py
from CommML.Linear_Models import linear_regression

x = [i for i in range(10)]
y = [2*j for j in x]

model = linear_regression()
model.fit(x,y,epochs=10)
model.predict(40)
```

## **Disclaimer** 
This software is a personal project and is intended for educational and non-commercial purposes only. Any commercial use or redistribution is strictly prohibited.

