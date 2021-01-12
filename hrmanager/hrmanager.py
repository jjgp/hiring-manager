# %%
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from hrmanager.schema import HIGH_PERFORMER_COL
from hrmanager.schema import INDEX_COL
from hrmanager.schema import PREDICTOR_COLS
from hrmanager.schema import PROTECTED_GROUP_COL
from hrmanager.schema import RETAINED_COL
from hrmanager.schema import TARGET_COLS

# %%
predictor_cols = list(PREDICTOR_COLS)
target_cols = list(TARGET_COLS)
usecols = [INDEX_COL] + target_cols + predictor_cols
train = pd.read_csv(
    "../data/train.csv",
    index_col="UNIQUE_ID",
    usecols=usecols,
    na_values=" ",
)
train.dropna(subset=target_cols, inplace=True)

X = train[predictor_cols]
y = train[target_cols]

# Create a new target that represents candidates who are high performers and
# retained
high_performer_retained_col = f"{HIGH_PERFORMER_COL}_{RETAINED_COL}"
y[high_performer_retained_col] = y.apply(
    lambda row: int(row[HIGH_PERFORMER_COL] > 0 and row[RETAINED_COL] > 0),
    axis=1,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# %%
estimator = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model = OneVsRestClassifier(estimator)

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer()),
        ("model", model),
    ],
)
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)

# %%
y_pred = y_proba > 0.5
print(HIGH_PERFORMER_COL)
print(accuracy_score(y_test[HIGH_PERFORMER_COL], y_pred[:, 0]))
print(confusion_matrix(y_test[HIGH_PERFORMER_COL], y_pred[:, 0]))
print(PROTECTED_GROUP_COL)
print(accuracy_score(y_test[PROTECTED_GROUP_COL], y_pred[:, 1]))
print(confusion_matrix(y_test[PROTECTED_GROUP_COL], y_pred[:, 1]))
print(RETAINED_COL)
print(accuracy_score(y_test[RETAINED_COL], y_pred[:, 2]))
print(confusion_matrix(y_test[RETAINED_COL], y_pred[:, 2]))
print(high_performer_retained_col)
print(accuracy_score(y_test[high_performer_retained_col], y_pred[:, 3]))
print(confusion_matrix(y_test[high_performer_retained_col], y_pred[:, 3]))

# %%
"""
Final_score = Overall_accuracy – Unfairness

Overall_accuracy=
Percentage_of_true_top_performers_hired * 25+
Percentage_of_true_retained_hired * 25 +
Percentage_of_true_retained_top_perf_hired * 50

Unfairness = Absolute_value(1 - Adverse_impact_ratio) * 100
"""

# The following is a scoring mechanism inspired by the above
# It is a heursitic and not the actual calculation
# The purpose is to augment the probabilities with a heuristic and sort them
assert target_cols == list(TARGET_COLS)
hr_scores = np.zeros((y_proba.shape[0]))
hr_scores = (
    hr_scores[:] + 0.25 * y_proba[:, 0] + 0.25 * y_proba[:, 2] + 0.5 * y_proba[:, 3]
)
hr_scores = hr_scores[:] + 0.1 * y_proba[:, 1]

hr_score_col = "HR_SCORE"
y_hired = y_test.copy()
y_hired[hr_score_col] = hr_scores
y_hired.sort_values(by=[hr_score_col], ascending=False, inplace=True)
y_hired = y_hired.head(y_hired.shape[0] // 2)

thp_count_test = y_test[y_test[HIGH_PERFORMER_COL] == 1.0].shape[0]
tr_count_test = y_test[y_test[RETAINED_COL] == 1.0].shape[0]
thpr_count_test = y_test[
    (y_test[HIGH_PERFORMER_COL] == 1.0) & (y_test[RETAINED_COL] == 1.0)
].shape[0]
protected_count_test = y_test[y_test[PROTECTED_GROUP_COL] == 1.0].shape[0]
non_protected_count_test = y_test[y_test[PROTECTED_GROUP_COL] == 0.0].shape[0]

thp_count_hired = y_hired[y_hired[HIGH_PERFORMER_COL] == 1.0].shape[0]
tr_count_hired = y_hired[y_hired[RETAINED_COL] == 1.0].shape[0]
thpr_count_hired = y_hired[
    (y_hired[HIGH_PERFORMER_COL] == 1.0) & (y_test[RETAINED_COL] == 1.0)
].shape[0]
protected_count_hired = y_hired[y_hired[PROTECTED_GROUP_COL] == 1.0].shape[0]
non_protected_count_hired = y_hired[y_hired[PROTECTED_GROUP_COL] == 0.0].shape[0]

adverse_impact_ratio = (protected_count_hired / protected_count_test) / (
    non_protected_count_hired / non_protected_count_test
)
unfairness = abs(1.0 - adverse_impact_ratio) * 100

print(unfairness)

print(
    25 * (thp_count_hired / thp_count_test)
    + 25 * (tr_count_hired / tr_count_test)
    + 50 * (thpr_count_hired / thpr_count_test)
    - unfairness,
)

# %%
