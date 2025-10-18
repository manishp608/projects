import pandas as pd
import numpy as np

df = [
    [35.0, 0.0],
    [42.0, 0.0],
    [50.0, 0.0],
    [60.0, 0.0],
    [67.0, 1.0],
    [75.0, 1.0],
    [80.0, 1.0],
    [90.0, 1.0],
    [95.0, 1.0],
    [100.0, 1.0],
    [110.0, 1.0],
    [120.0, 1.0],
    [130.0, 1.0],
    [140.0, 1.0],
    [150.0, 1.0],
    [160.0, 1.0],
    [175.0, 1.0],
    [190.0, 1.0],
    [210.0, 1.0],
    [230.0, 1.0]
]

df = pd.DataFrame(df, columns=['exam_score', 'admitted'])

def sigmoid(z):
    return 1 / (np.exp(-z))

def loss_fn(X, y, theta0, theta1):
    ep = 1e-8
    n = len(X)
    z = theta0 + theta1 * X
    y_cap = sigmoid(z)

    grad = np.sum((y * np.log(y_cap + ep)) + ((1 - y) * np.log(1 - y_cap + ep)))
    return (-1 / n) * grad

def gd(X, y, alpha, epochs):
    theta0, theta1 = 0, 0
    n = len(X)

    for i in range(epochs):
        z = theta0 + theta1 * X
        y_cap = sigmoid(z)

        grad0 = np.sum(y_cap - y) / n
        grad1 = np.sum((y_cap - y) * X) / n

        theta0 -= alpha * grad0
        theta1 -= alpha * grad1

        return theta0, theta1
    
