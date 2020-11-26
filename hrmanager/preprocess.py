# %%
import pandas as pd

df = pd.read_csv("../data/train.csv", index_col="UNIQUE_ID")
criterion_df = df.loc[:, "Overall_Rating":"Retained"]
predictor_df = df.loc[:, "SJ_Most_1":"PScale13_Q5"]

# %%
numeric_cols = ["Scenario1_Time", "Scenario2_Time"] \
    + [f"SJ_Time_{number}" for number in range(1, 10)]
binary_features_df = predictor_df.drop(numeric_cols, axis=1)
numeric_features_df = predictor_df.loc[:, numeric_cols]

# %%
