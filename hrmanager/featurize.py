from schema import BINARY_COLS, NUMERIC_COLS
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def featurize(df):
    binary_features_matrix = OneHotEncoder().fit_transform(
        df[list(BINARY_COLS)]
    )
    numeric_features_df = df[list(NUMERIC_COLS)]
    X = np.concatenate(
        (binary_features_matrix.toarray(), numeric_features_df.to_numpy()),
        axis=1
    )
    # NOTE: for now Y is just the "Retained" column!
    Y = df["Retained"].to_numpy()
    return X, Y
