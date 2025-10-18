import pandas as pd
import numpy as np
import math

df = [
    ["I love this movie", 1],
    ["This film was great", 1],
    ["Amazing acting and story", 1],
    ["What a fantastic performance", 1],
    ["The direction was brilliant", 1],
    ["Highly enjoyable and emotional", 1],
    ["I really liked the soundtrack", 1],
    ["The characters were inspiring", 1],
    ["This was a boring movie", 0],
    
    ["I did not like the film", 0],
    ["Terrible acting and weak story", 0],
    ["The performance was disappointing", 0],
    ["It was a waste of time", 0],
    ["Not enjoyable at all", 0],
    ["I hated the soundtrack", 0],
    ["The characters were dull", 0],
    ["Amazing visuals but boring story", 0],
    ["Loved the emotional moments", 1],
    ["Brilliant soundtrack and acting", 1],
    ["Terrible direction and slow pace", 0],
    ["Enjoyable family film", 1],
    ["Weak characters and dull ending", 0],
    ["Fantastic film overall", 1],
    ["Not worth watching", 0]
]
df = pd.DataFrame(df, columns=['Text', 'Label'])

test_ratio = 0.3
split_idx = int(len(df) * (1 - test_ratio))
train_df = df[:split_idx]
test_df = df[split_idx:]

counts = {0: 0, 1: 0}
wfreq = {0: {}, 1:{}}
word_set = set()


for sen, cat in train_df.values:
    counts[cat] += 1
    for word in sen.split():
        word_set.add(word)
        wfreq[cat][word] = wfreq[cat].get(word, 0) + 1
doc_count = len(train_df)

p = {x: counts[x] / doc_count for x in [0, 1]}
total_words = len(word_set)

pword = {0: {}, 1: {}}
for x in [0, 1]:
    total_word_count = sum(wfreq[x].values())
    for word in word_set:
        pword[x][word] = (wfreq[x].get(word, 0) + 1) / (total_word_count + total_words)



def classify(sen):
    tokens = sen.split()
    results = {}
    for x in [0, 1]:
        score = np.log(p[x])
        for token in tokens:
            if token in word_set:
                score += math.log(pword[x][token])
        results[x] = score

    return max(results, key=results.get)

predi = []
act = []
for sen, cat in test_df.values:
    pred = classify(sen.lower())
    predi.append(pred)
    act.append(cat)

predi = np.array(predi)
act = np.array(act)

tp = np.sum((predi == 1) & (act == 1))
tn = np.sum((predi == 0) & (act == 0))
fn = np.sum((predi == 0) & (act == 1))
fp = np.sum((predi == 1) & (act == 0))

acc = (tp + tn) / len(act)
pre = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = (2 * pre * rec) / (pre + rec)
