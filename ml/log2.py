import pandas as pd
import numpy as np

df = [
    [35, 40, 5, 0],
    [42, 50, 6, 0],
    [50, 52, 7, 0],
    [60, 65, 8, 0],
    [67, 70, 9, 1],
    [75, 78, 10, 1],
    [80, 85, 12, 1],
    [90, 88, 14, 1],
    [95, 90, 15, 1],
    [100, 92, 16, 1],
    [110, 100, 17, 1],
    [120, 105, 18, 1],
    [130, 110, 19, 1],
    [140, 115, 20, 1],
    [150, 118, 22, 1],
    [160, 120, 24, 1],
    [175, 125, 25, 1],
    [190, 128, 26, 1],
    [210, 130, 28, 1],
    [230, 135, 30, 1]
]

df = pd.DataFrame(df, columns=['exam1', 'exam2', 'hours_study', 'admitted'])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_fn(X, y, theta):
    z = X.dot(theta)
    y_cap = sigmoid(z)
    ep = 1e-8
    n = len(X)

    return -np.sum((y * np.log(y_cap + ep)) + ((1 - y) * np.log(1 - y_cap + ep))) / n

def gd(X, y, alpha, epochs):
    theta = np.zeros(X.shape[1])
    n = len(X)

    for i in range(epochs):
        z = X.dot(theta)
        y_cap = sigmoid(z)

        grad = X.T.dot(y_cap - y) / n
        theta -= alpha * grad

    return theta

X_aug = np.c_[np.ones(X.shape[0]), X]

