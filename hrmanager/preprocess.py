# %%
import pandas as pd

# %%
df = pd.read_csv("../data/train.csv", index_col="UNIQUE_ID")

# %%
df.head(1)
df.Teamwork.unique()

# %%
