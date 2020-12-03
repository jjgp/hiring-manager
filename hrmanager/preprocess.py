# %%
from schema import BINARY_COLS, BINARY_CRITERION_COLS, BINARY_PREDICTOR_COLS, \
    INDEX_COL, NUMERIC_COLS, NUMERIC_CRITERION_COLS, NUMERIC_PREDICTOR_COLS, \
    TARGET_COLS
import pandas as pd


def predictors_dataset(filepath, target_cols=list(TARGET_COLS), dropna=False):
    binary_cols = list(BINARY_PREDICTOR_COLS)
    numeric_cols = list(NUMERIC_PREDICTOR_COLS)
    df = read_csv(filepath, target_cols + binary_cols + numeric_cols)
    # NOTE: filling NaN with 0.0s
    df[binary_cols] = df[binary_cols].fillna(0)
    # for col in numeric_cols:
    #     mode = df[col].mode(dropna=True).iloc[-1]
    #     # NOTE: filling NaN with the mode of that column
    #     df[col] = df[col].fillna(mode)
    return df.dropna() if dropna else df


def criterion_dataset(filepath, target_cols=list(TARGET_COLS), dropna=False):
    df = read_csv(filepath, target_cols +
                  list(BINARY_CRITERION_COLS + NUMERIC_CRITERION_COLS))
    return df.dropna() if dropna else df


def read_csv(filepath, usecols=list(TARGET_COLS + BINARY_COLS + NUMERIC_COLS)):
    usecols = [INDEX_COL] + usecols
    return pd.read_csv(
        filepath,
        index_col=INDEX_COL,
        usecols=usecols,
        na_values=" "
    )


# # %%
# train_csv = "../data/train.csv"
# predictors_dataset(train_csv, dropna=True)

# # %%
# criterion_dataset(train_csv, dropna=True)
