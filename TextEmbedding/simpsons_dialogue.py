import pandas as pd
from collections import Counter
df = pd.read_csv("datasets/simpsons_dataset.csv")
print(df.shape)
counts = Counter(df["raw_character_text"])
print(counts.most_common(5))