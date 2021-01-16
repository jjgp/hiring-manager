# %%
import numpy as np
import pandas as pd
from schema import HIGH_PERFORMER_COL
from schema import INDEX_COL
from schema import PREDICTOR_COLS
from schema import PROTECTED_GROUP_COL
from schema import RETAINED_COL
from schema import TARGET_COLS
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# %% [markdown]
# # Extracting targets and predictors from train.csv

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

# %% [markdown]
# # Adding derived targets
# 1. _High performer retained_ candidates who are high performers and
# retained
# 2. _Exclusive retained_ candidates that are only retained
# 3. _Exclusive high performer_ candidates that are only high performer

# %%
derived_targets_map = {
    f"{HIGH_PERFORMER_COL}_{RETAINED_COL}": lambda row: int(
        row[HIGH_PERFORMER_COL] > 0 and row[RETAINED_COL] > 0,
    ),
    f"Exclusive_{RETAINED_COL}": lambda row: int(
        row[HIGH_PERFORMER_COL] < 1 and row[RETAINED_COL] > 0,
    ),
    f"Exclusive_{HIGH_PERFORMER_COL}": lambda row: int(
        row[HIGH_PERFORMER_COL] > 0 and row[RETAINED_COL] < 1,
    ),
}

for derived_target, derivation in derived_targets_map.items():
    y[derived_target] = y.apply(derivation, axis=1)
    target_cols.append(derived_target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# %% [markdown]
# # Pipeline and training

# %%
estimator = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
)
model = OneVsRestClassifier(estimator)

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", model),
    ],
)
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)

# %% [markdown]
# # Evaluation

# %%
y_pred = y_proba > 0.5
for idx, target in enumerate(target_cols):
    print(f"{target}")
    print("-" * len(target))
    print(f"accuracy: {accuracy_score(y_test[target], y_pred[:, idx])}")
    print(f"confusion matrix:\n{confusion_matrix(y_test[target], y_pred[:, idx])}")
    print()

# %% [markdown]
# # Ranking hires based on predictions and heuristic

# %%
# This heuristic is based on the actual scoring mechanism
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

# %% [markdown]
# # Final score

# %%
def extract_counts(split):  # noqa: E302
    thp_count = split[split[HIGH_PERFORMER_COL] == 1.0].shape[0]
    tr_count = split[split[RETAINED_COL] == 1.0].shape[0]
    thpr_count = split[
        (split[HIGH_PERFORMER_COL] == 1.0) & (split[RETAINED_COL] == 1.0)
    ].shape[0]
    pro_count = split[split[PROTECTED_GROUP_COL] == 1.0].shape[0]
    priv_count = split[split[PROTECTED_GROUP_COL] == 0.0].shape[0]
    return (thp_count, tr_count, thpr_count, pro_count, priv_count)


(
    thp_count_hired,
    tr_count_hired,
    thpr_count_hired,
    pro_count_hired,
    priv_count_hired,
) = extract_counts(y_hired)

(
    thp_count_test,
    tr_count_test,
    thpr_count_test,
    pro_count_test,
    priv_count_test,
) = extract_counts(y_test)

"""
Final_score = Overall_accuracy – Unfairness

Overall_accuracy=
Percentage_of_true_top_performers_hired * 25+
Percentage_of_true_retained_hired * 25 +
Percentage_of_true_retained_top_perf_hired * 50

Unfairness = Absolute_value(1 - Adverse_impact_ratio) * 100
"""
ratio_pro = pro_count_hired / pro_count_test
print(f"ratio protected hired: {ratio_pro}")

ratio_priv = priv_count_hired / priv_count_test
print(f"ratio privilegded hired: {ratio_priv}")

adverse_impact_ratio = ratio_pro / ratio_priv
print(f"adverse impact ratio: {adverse_impact_ratio}")

unfairness = abs(1.0 - adverse_impact_ratio) * 100
print(f"unfairness: {unfairness}")

ratio_thp = thp_count_hired / thp_count_test
print(f"ratio true high performer: {ratio_thp}")

ratio_tr = tr_count_hired / tr_count_test
print(f"ratio true retained: {ratio_tr}")

ratio_thpr = thpr_count_hired / thpr_count_test
print(f"ratio true high performer retained: {ratio_thpr}")

final_score = 25 * ratio_thp + 25 * ratio_tr + 50 * ratio_thpr - unfairness
print(f"final score: {final_score}")

# %%
