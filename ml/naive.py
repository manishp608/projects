import pandas as pd
import numpy as np

df = [
    [4, 122, 61, 7, 52, 27.82219568, 0.635762173, 48, 0, 0],
    [1, 104, 56, 11, 0, 24.24503026, 0.089848385, 60, 0, 0],
    [3, 109, 62, 23, 0, 18.87690204, 0.332317171, 61, 0, 0],
    [3, 117, 87, 21, 108, 30.4921571, 0.405407555, 25, 1, 0],
    [2, 114, 56, 28, 10, 23.1479042, 0.120255598, 25, 0, 0],
    [3, 120, 95, 13, 165, 32.39493617, 0.211376261, 26, 1, 0],
    [2, 99, 59, 20, 3, 44.38536126, 0.219177313, 43, 0, 0],
    [3, 150, 78, 13, 116, 33.39655129, 0.309157203, 23, 0, 0],
    [0, 122, 68, 27, 19, 45.98096352, 0.298969424, 57, 0, 0],
    [2, 104, 66, 31, 110, 25.12408723, 0.494678442, 53, 0, 0],
    [4, 130, 54, 24, 0, 23.0334144, 0.340692543, 63, 1, 0],
    [2, 117, 68, 30, 43, 24.81089088, 0.260540091, 41, 0, 0],
    [1, 67, 56, 20, 12, 31.54429623, 0.116002259, 40, 0, 0],
    [3, 130, 49, 28, 150, 51.06064081, 0.081957536, 21, 0, 0],
    [4, 89, 71, 20, 116, 20.36519701, 0.105820288, 59, 1, 0],
    [4, 148, 62, 9, 0, 36.33764183, 0.559926645, 67, 1, 0],
    [1, 112, 90, 22, 63, 27.10024265, 0.679502021, 37, 1, 0],
    [2, 143, 82, 10, 18, 30.43337864, 0.145064816, 49, 0, 0],
    [3, 101, 90, 43, 156, 23.19386062, 0.165994, 28, 1, 0],
    [2, 122, 61, 16, 181, 38.15396827, 0.491571652, 46, 1, 0],
    [3, 156, 71, 26, 0, 31.69252407, 0.039152253, 48, 0, 0],
    [4, 111, 38, 19, 146, 39.61217485, 0.278828256, 54, 0, 0],
    [1, 102, 66, 7, 14, 33.22671636, 0.462085911, 46, 0, 0],
    [3, 139, 71, 7, 59, 18.47863216, 0.090168205, 33, 0, 0],
    [3, 139, 71, 27, 82, 39.67579957, 0.238599802, 64, 0, 0],
    [3, 103, 69, 35, 225, 17.90899557, 0.501099654, 57, 0, 0],
    [5, 60, 70, 12, 0, 18.91068122, 0.155102847, 33, 1, 0],
    [0, 101, 37, 7, 146, 25.57958124, 0.096417867, 58, 0, 0],
    [1, 140, 73, 16, 0, 25.67301121, 0.255078432, 45, 1, 0],
    [4, 96, 75, 10, 0, 41.77214956, 0.376397231, 67, 0, 0],
]

df = pd.DataFrame(df, columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
    "Outlier"
]
)

test_ratio = 0.3
split_idx = int(len(df) * (1 - test_ratio))
train_df = df[:split_idx]
test_df = df[split_idx:]

train_X = train_df.iloc[:, :-2]
train_y = train_df.iloc[:, -2]

test_X = test_df.iloc[:, :-2]
test_y = test_df.iloc[:, -2]

def gaussian_prob(X, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * (np.exp(-0.5 * (((X - mean) / std) ** 2)))

def train_bayes(train_df):
    n, features = train_df.shape[0], train_df.shape[1] - 2
    class_0_df = train_df[train_df[:, -2] == 0]
    class_1_df = train_df[train_df[:, -2] == 1]

    prior_0 = len(class_0_df) / len(train_df)
    prior_1 = len(class_1_df) / len(train_df)

    means_0 = class_0_df.iloc[:, :-2].mean()
    means_1 = class_1_df.iloc[:, :-2].mean()

    std_0 = class_0_df.iloc[:, :-2].std(ddof=0)   
    std_1 = class_1_df.iloc[:, :-2].std(ddof=0) 

    return prior_0, prior_1, means_0, std_0, means_1, std_1

def predict(test_df, prior_0, prior_1, means_0, std_0, means_1, std_1):
    df = test_df.iloc[:, :-2]
    prob_0 = gaussian_prob(df, means_0, std_0)
    prob_1 = gaussian_prob(df, means_1, std_1)

    like_0 = prob_0.prod(axis=1) * prior_0
    like_1 = prob_1.prod(axis=1) * prior_1

    pred = (like_1 > like_0).astype(int)

    return pred

def metrics(pred, act):
    tp = np.sum((pred == 1) & (act == 1))
    tn = np.sum((pred == 0) & (act == 0))
    fn = np.sum((pred == 0) & (act == 1))
    fp = np.sum((pred == 1) & (act == 0))

    acc = (tp + tn) / len(act)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * pre * rec) / (pre + rec)

    return acc, pre, rec, f1

prior_0, prior_1, means_0, std_0, means_1, std_1 = train_bayes(train_df)
pred = predict(test_df, prior_0, prior_1, means_0, std_0, means_1, std_1)
acc, pre, rec, f1 = metrics(pred, test_y)