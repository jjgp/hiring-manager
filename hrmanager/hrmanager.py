# %%
from schema import BINARY_COLS, NUMERIC_COLS
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# %%
def preprocess_csv(filepath):  # noqa: E302
    df = pd.read_csv(filepath, index_col="UNIQUE_ID", na_values=" ")
    # NOTE: filling NaN with 0.0s
    df[list(BINARY_COLS)] = df[list(BINARY_COLS)].fillna(0)
    for col in NUMERIC_COLS:
        mode = df[col].mode(dropna=True).iloc[-1]
        # NOTE: filling NaN with the mode of that column
        df[col] = df[col].fillna(mode)
    return df


# %%
def featurize(df):  # noqa: E302
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


# %%
df = preprocess_csv("../data/train.csv")
df.head()

# %%
X, Y = featurize(df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

estimator = ExtraTreesClassifier(random_state=0)
estimator.fit(X_train, Y_train)
Y_pred = estimator.predict(X_test)

# %%
confusion_matrix(Y_test, Y_pred)

# %%
accuracy_score(Y_test, Y_pred)

# %%
importances = estimator.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in estimator.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# %%
