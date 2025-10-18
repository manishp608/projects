import pandas as pd
import numpy as np

df = [
    [35.0, 179.0],
    [42.0, 200.0],
    [50.0, 221.0],
    [60.0, 263.0],
    [67.0, 280.0],
    [75.0, 314.0],
    [80.0, 327.0],
    [90.0, 360.0],
    [95.0, 377.0],
    [100.0, 391.0],
    [110.0, 425.0],
    [120.0, 462.0],
    [130.0, 493.0],
    [140.0, 521.0],
    [150.0, 552.0],
    [160.0, 582.0],
    [175.0, 631.0],
    [190.0, 675.0],
    [210.0, 740.0],
    [230.0, 804.0]
]
df = pd.DataFrame(df, columns=['feature', 'target'])

def gd(X, y, alpha, epochs):
    theta0, theta1 = 0, 0
    n = len(X)

    for i in range(epochs):
        y_cap = theta0 + theta1 * X
        grad0 = sum(y_cap - y) / n
        grad1 = sum((y_cap - y) * X) / n

        theta0 -= alpha * grad0
        theta1 -= alpha * grad1
    return theta0, theta1

theta0, theta1 = 0, 0

def mse(X, y):
    y_cap = theta0 + theta1 * X
    n = len(X)
    return sum((y_cap - y) ** 2) / (2 * n)

