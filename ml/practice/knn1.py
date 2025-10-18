import pandas as pd
import numpy as np
import math
from collections import Counter

df = [
    [5.1, 3.5, 1.4, 0.2, "setosa"],
    [4.9, 3.0, 1.4, 0.2, "setosa"],
    [4.7, 3.2, 1.3, 0.2, "setosa"],
    [4.6, 3.1, 1.5, 0.2, "setosa"],
    [5.0, 3.6, 1.4, 0.2, "setosa"],
    [5.4, 3.9, 1.7, 0.4, "setosa"],
    [5.8, 4.0, 1.2, 0.2, "setosa"],
    [6.0, 2.2, 4.0, 1.5, "versicolor"],
    [6.1, 2.8, 4.7, 1.4, "versicolor"],
    [5.9, 3.0, 4.2, 1.5, "versicolor"],
    [6.7, 3.1, 4.4, 1.4, "versicolor"],
    [6.3, 2.5, 4.9, 1.5, "versicolor"],
    [6.5, 3.0, 5.1, 2.0, "virginica"],
    [6.2, 2.8, 4.5, 1.5, "versicolor"],
    [6.4, 2.9, 4.3, 1.3, "versicolor"],
    [5.5, 2.4, 4.0, 1.3, "versicolor"],
    [5.7, 2.8, 4.1, 1.3, "versicolor"],
    [5.8, 2.7, 5.1, 1.9, "virginica"],
    [6.9, 3.1, 5.4, 2.3, "virginica"],
    [6.0, 2.2, 5.0, 1.5, "virginica"],
    [6.3, 2.3, 5.6, 2.4, "virginica"],
    [6.1, 2.8, 5.6, 2.4, "virginica"],
    [5.6, 2.9, 3.6, 1.3, "versicolor"],
    [5.8, 2.7, 4.1, 1.0, "versicolor"],
    [6.0, 2.9, 4.5, 1.5, "versicolor"],
    [6.1, 2.6, 4.7, 1.4, "versicolor"],
    [6.5, 3.0, 5.2, 2.0, "virginica"],
    [6.2, 2.9, 5.4, 2.3, "virginica"],
    [5.9, 3.0, 5.1, 1.8, "virginica"],
    [6.3, 2.7, 5.6, 2.1, "virginica"]
]
df = pd.DataFrame(df, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWdith', 'Class'])

def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

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

def cross_validate(df, k_values, folds=5):
    results = {}
    split_data = k_fold_split(df, folds)
    for k in k_values:
        acc_list = []
        for i in range(folds):
            test_df = split_data[i]
            train_df = pd.concat([split_data[j] for j in range(folds) if j != i])
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
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    train_X, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    test_X, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    correct = 0
    for idx in range(len(test_df)):
        pred = knn(train_X, train_y, test_X.iloc[idx], best_k)
        if pred == test_y.iloc[idx]:
            correct += 1
        
    accuracy = round(correct / len(test_df), 2)
    print(f"k={best_k} Accuracy:{accuracy:.2f}")