import pandas as pd
import numpy as np
import math
from collections import Counter

df = [
    [5.2, 3.4, 1.5, 0.2, "setosa"],
    [4.8, 3.1, 1.6, 0.3, "setosa"],
    [5.0, 3.2, 1.2, 0.2, "setosa"],
    [5.3, 3.7, 1.4, 0.3, "setosa"],
    [4.9, 3.0, 1.5, 0.1, "setosa"],
    [5.1, 3.5, 1.3, 0.3, "setosa"],
    [5.4, 3.4, 1.7, 0.2, "setosa"],
    [5.0, 3.3, 1.4, 0.2, "setosa"],
    [6.0, 2.7, 4.2, 1.3, "versicolor"],
    [6.2, 2.9, 4.3, 1.3, "versicolor"],
    [5.7, 2.6, 3.5, 1.0, "versicolor"],
    [5.8, 2.7, 4.1, 1.2, "versicolor"],
    [6.1, 3.0, 4.6, 1.4, "versicolor"],
    [5.6, 2.8, 4.0, 1.3, "versicolor"],
    [6.3, 2.5, 4.9, 1.5, "versicolor"],
    [6.0, 3.4, 4.5, 1.6, "versicolor"],
    [5.9, 3.0, 4.2, 1.5, "versicolor"],
    [6.4, 2.8, 5.0, 1.7, "versicolor"],
    [5.5, 2.5, 4.0, 1.2, "versicolor"],
    [6.2, 2.2, 4.8, 1.8, "versicolor"],
    [6.5, 3.0, 5.2, 2.0, "virginica"],
    [6.9, 3.1, 5.4, 2.1, "virginica"],
    [6.7, 3.0, 5.8, 2.2, "virginica"],
    [7.1, 3.0, 5.9, 2.1, "virginica"],
    [6.3, 2.9, 5.6, 1.8, "virginica"],
    [6.6, 2.8, 5.3, 2.0, "virginica"],
    [7.0, 3.2, 5.7, 2.3, "virginica"],
    [6.5, 3.2, 5.1, 2.0, "virginica"],
    [6.8, 3.0, 5.5, 2.1, "virginica"],
    [6.4, 2.9, 5.6, 2.2, "virginica"],
    [6.2, 3.4, 5.4, 2.3, "virginica"],
    [6.9, 3.1, 5.1, 2.3, "virginica"],
    [7.2, 3.2, 6.0, 2.2, "virginica"],
    [6.3, 2.8, 5.7, 1.9, "virginica"],
    [6.1, 3.0, 5.5, 1.8, "virginica"],
    [6.7, 3.3, 5.7, 2.1, "virginica"],
    [6.4, 3.1, 5.5, 1.8, "virginica"],
    [6.8, 3.2, 5.9, 2.3, "virginica"],
    [7.3, 2.9, 6.1, 2.5, "virginica"],
    [6.5, 3.0, 5.8, 2.2, "virginica"]
]
df = pd.DataFrame(df, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class'])

def distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def k_fold_split(df, k):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return np.array_split(df, k)

def knn(train_X, train_y, test_x, k):
    distances = []
    for i in range(len(train_X)):
        dist = distance(train_X.iloc[i], test_x)
        distances.append((dist, train_y.iloc[i]))
    distances.sort(key=lambda x: x[0])
    top_k = [label for _, label in distances[:k]]
    return Counter(top_k).most_common(1)[0][0]

def normalize(df):
    df_norm = df.copy()
    for col in df_norm.columns[:-1]:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df_norm

def cross_validate(df, k_values, folds=5):
    results = {}
    split_df = k_fold_split(df, folds)
    for k in k_values:
        acc_list = []
        for i in range(folds):
            test_df = split_df[i]
            train_df = pd.concat([split_df[j] for j in range(folds) if j != i])
            train_X, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
            test_X, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
            correct = 0
            for idx in range(len(test_df)):
                pred = knn(train_X, train_y, test_X.iloc[idx], k)
                if pred == test_y.iloc[idx]:
                    correct += 1 
            acc_list.append(correct / len(test_df))
        results[k] = round(np.mean(acc_list), 4)

    return results

def final_test(df, best_k, test_ratio=0.2):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df, test_df = df[:split_idx], df[split_idx:]
    train_X, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    test_X, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    correct = 0
    for idx in range(len(test_df)):
        pred = knn(train_X, train_y, test_X.iloc[idx], best_k)
        if pred == test_y.iloc[idx]:
            correct += 1
    accuracy = round(correct / len(test_df), 2)
    print(f"k: {best_k} Accuracy: {accuracy:.2f}")    