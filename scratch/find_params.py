import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("data/train_phase1.csv")
df_eval = pd.read_csv("data/eval.csv")

X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]
X_eval = df_eval.drop(columns=["target"])
y_eval = df_eval["target"]

for n in [50, 100, 200, 300, 500]:
    for md in [None, 5, 10, 15, 20, 30]:
        for ms in [2, 5, 10]:
            model = RandomForestClassifier(n_estimators=n, max_depth=md, min_samples_split=ms, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            acc = accuracy_score(y_eval, preds)
            if acc >= 0.7:
                print(f"FOUND: n={n}, md={md}, ms={ms} -> acc={acc}")
            elif acc > 0.69:
                 print(f"Close: n={n}, md={md}, ms={ms} -> acc={acc}")
print("Done")
