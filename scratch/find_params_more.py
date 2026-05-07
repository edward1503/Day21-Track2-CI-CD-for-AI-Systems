import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("data/train_phase1.csv")
df_eval = pd.read_csv("data/eval.csv")

X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]
X_eval = df_eval.drop(columns=["target"])
y_eval = df_eval["target"]

for crit in ["gini", "entropy"]:
    for cw in [None, "balanced"]:
        for mf in ["sqrt", "log2", None]:
            model = RandomForestClassifier(n_estimators=300, criterion=crit, class_weight=cw, max_features=mf, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_eval)
            acc = accuracy_score(y_eval, preds)
            print(f"crit={crit}, cw={cw}, mf={mf} -> acc={acc}")
