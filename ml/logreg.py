import numpy as np
import pandas as pd


data = [
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


df = pd.DataFrame(data, columns=['exam_score', 'admitted'])




print("First 5 rows:")
print(df.head(), "\n")

print(f"Shape (N, d): {df.shape} \n")

print("Summary statistics for exam_score:")
print(f"Min: {df['exam_score'].min()}")
print(f"Max: {df['exam_score'].max()}")
print(f"Mean: {df['exam_score'].mean():.2f}")
print(f"Std: {df['exam_score'].std(ddof=0):.2f}\n")


def calculate_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_loss(y_true, y_pred):
    ep = 1e-8
    return -np.mean(y_true * np.log(y_pred + ep) + (1 - y_true) * np.log(1 - y_pred + ep))


X = df['exam_score'].values
y = df['admitted'].values
N = len(y)

t0 = 0.0
t1 = 0.0
a = 0.01
epochs = 1000

for epoch in range(epochs):
    z = t0 + t1 * X
    y_pred = calculate_sigmoid(z)

    g_theta0 = np.mean(y_pred - y)
    g_theta1 = np.mean((y_pred - y) * X)

    t0 -= a * g_theta0
    t1 -= a * g_theta1


final_ypreds = calculate_sigmoid(t0 + t1 * X)
final_yloss = calculate_loss(y, final_ypreds)


print(f"Final theta0: {t0:.2f}")
print(f"Final theta1: {t1:.2f}")
print(f"Final loss: {final_yloss:.2f} \n")



def predict_score(score):
    return calculate_sigmoid(t0 + t1 * score)

print(f"Prediction for exam_score=65: {predict_score(65):.2f}")
print(f"Prediction for exam_score=155: {predict_score(155):.2f}")


