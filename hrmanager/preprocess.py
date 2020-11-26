# %%
import pandas as pd

# Criterion
criterion_cols = [
    "Overall_Rating",
    "Technical_Skills",
    "Teamwork",
    "Customer_Service",
    "Hire_Again",
    "High_Performer",
    "Retained",
    "Protected_Group"
]

# Biodata/Work History Items
biodata_cols = [f"Biodata_{num:02}" for num in range(1, 21)]

# Personality/Work Style Items
pscale_cols = [
    f"PScale{first:02}_Q{second}"
    for first in range(1, 14)
    for second in range(1, 5)
]
pscale_cols.extend(["PScale06_Q5", "PScale06_Q6", "PScale13_Q5"])

# Scenario Interpretation
scenario_cols = [
    f"Scenario{first}_{second}"
    for first in ["1", "2"]
    for second in ["1", "2", "3", "4", "5", "6", "7", "8", "Time"]
]

# Situational Judgment Items
sj_cols = [
    f"SJ_{first}_{second}"
    for first in ["Least", "Most", "Time"]
    for second in range(1, 10)
]

# Features
feature_cols = biodata_cols \
    + pscale_cols \
    + scenario_cols \
    + sj_cols

use_cols = ["UNIQUE_ID"] + criterion_cols + feature_cols

# %%
df = pd.read_csv("../data/train.csv", index_col="UNIQUE_ID", usecols=use_cols)

# %%
