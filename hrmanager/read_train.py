# %%
import pandas as pd
from schema import INDEX_COL
from schema import PREDICTOR_COLS

# %%
def read_train(target_cols):  # noqa: E302
    predictor_cols = list(PREDICTOR_COLS)
    usecols = [INDEX_COL] + target_cols + predictor_cols
    return pd.read_csv(
        "../data/train.csv",
        index_col="UNIQUE_ID",
        usecols=usecols,
        na_values=" ",
    )


# %%
