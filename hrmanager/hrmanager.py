# %%
from preprocess import predictors_dataset
from schema import BINARY_PREDICTOR_COLS, NUMERIC_PREDICTOR_COLS, \
    RETAINED_COL
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def featurize(df, binary_cols, numeric_cols, target_col):
    binary_features_matrix = OneHotEncoder().fit_transform(df[binary_cols])
    numeric_features_df = df[numeric_cols]
    X = np.concatenate((binary_features_matrix.toarray(),
                        numeric_features_df.to_numpy()), axis=1)
    # NOTE: for now Y is just the "Retained" column!
    Y = df[target_col].to_numpy()
    return X, Y


# %%
df = predictors_dataset("../data/train.csv")
df.head()

# %%
X, Y = featurize(df, list(BINARY_PREDICTOR_COLS),
                 list(NUMERIC_PREDICTOR_COLS), RETAINED_COL)
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
    [tree.feature_importances_ for tree in estimator.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# %%
