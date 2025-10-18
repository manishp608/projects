import pandas as pd
import numpy as np

df = [
    [35.0, 1.0, 20.0, 179.0],
    [42.0, 2.0, 15.0, 200.0],
    [50.0, 2.0, 18.0, 221.0],
    [60.0, 3.0, 10.0, 263.0],
    [67.0, 3.0, 8.0, 280.0],
    [75.0, 3.0, 12.0, 314.0],
    [80.0, 4.0, 5.0, 327.0],
    [90.0, 4.0, 5.0, 360.0],
    [95.0, 4.0, 6.0, 377.0],
    [100.0, 5.0, 5.0, 391.0],
    [110.0, 5.0, 3.0, 425.0],
    [120.0, 5.0, 2.0, 462.0],
    [130.0, 6.0, 2.0, 493.0],
    [140.0, 6.0, 1.0, 521.0],
    [150.0, 6.0, 1.0, 552.0],
    [160.0, 7.0, 1.0, 582.0],
    [175.0, 7.0, 2.0, 631.0],
    [190.0, 8.0, 2.0, 675.0],
    [210.0, 8.0, 1.0, 740.0],
    [230.0, 9.0, 1.0, 804.0]
]
df = pd.DataFrame(df, columns=['size', 'rooms', 'age', 'price'])

def gd(X, y, theta, alpha, epochs):
    theta = np.zeros(X.shape[1])
    n = len(X)

    for i in range(epochs):
        y_cap = X.dot(theta)
        grad = X.T.dot(y_cap - y) / n
        theta -= alpha * grad

    return theta

X_aug = np.c_[np.ones(X.shape[0]), X]

def mse(X, y, theta):
    n = len(X)
    y_cap = X.dot(theta)
    error = np.sum((y_cap - y) ** 2) / (2 * n)
    return error