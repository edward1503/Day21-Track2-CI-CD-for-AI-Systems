import pandas as pd
df = pd.read_csv("data/train_phase1.csv")
print(df["target"].value_counts(normalize=True))
