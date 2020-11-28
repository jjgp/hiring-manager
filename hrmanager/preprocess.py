from schema import BINARY_COLS, NUMERIC_COLS
import pandas as pd


def preprocess_csv(filepath):
    df = pd.read_csv(filepath, index_col="UNIQUE_ID", na_values=" ")
    # NOTE: filling NaN with 0.0s
    df[list(BINARY_COLS)] = df[list(BINARY_COLS)].fillna(0)
    for col in NUMERIC_COLS:
        mode = df[col].mode(dropna=True).iloc[-1]
        # NOTE: filling NaN with the mode of that column
        df[col] = df[col].fillna(mode)
    return df
