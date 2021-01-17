# %%
import numpy as np
import pandas as pd
from IPython.display import display
from schema import HIGH_PERFORMER_COL
from schema import INDEX_COL
from schema import PREDICTOR_COLS
from schema import PROTECTED_GROUP_COL
from schema import RETAINED_COL
from schema import TARGET_COLS
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# %% [markdown]
# # Extracting predictors and targets from train.csv

# %%
label_col = "label"
predictor_cols = list(PREDICTOR_COLS)
target_cols = list(TARGET_COLS)
usecols = [INDEX_COL] + target_cols + predictor_cols
weight_col = "weight"

train = pd.read_csv(
    "../data/train.csv",
    index_col="UNIQUE_ID",
    usecols=usecols,
    na_values=" ",
)
train.dropna(subset=target_cols, inplace=True)

X = train[predictor_cols]


# %% [markdown]
# # Creating class labels
#
# (None, 0), (High_Performer, 1), (Retained, 2), (High_Performer & Retained, 3)

# %%

targets = train[target_cols]
targets[label_col] = targets.apply(
    lambda row: int(row[HIGH_PERFORMER_COL] + 2 * row[RETAINED_COL]),
    axis=1,
)
targets.head()

# %% [markdown]
# # Train test split

# %%
X_train, X_test, targets_train, targets_test = train_test_split(
    X,
    targets,
    test_size=0.1,
    stratify=targets[label_col],
)
y_train, y_test = targets_train[label_col], targets_test[label_col]

# %% [markdown]
# # Computing sample weights
#
# The sample weights are based on the ratio of protected / priviledged classes. It's
# possible to do weighting based on the occurrences of the aforementioned in
# combination with the class label; however, left for futher investigation. Weights are
# calculated on the train set to avoid data leakage from the test set.

# %%
w_train = compute_sample_weight("balanced", targets_train[PROTECTED_GROUP_COL])
display(targets_train.head())
display(w_train[:5])

# %% [markdown]
# # Pipeline and training

# %%
estimator = XGBClassifier(
    objective="multiclass:softmax",
    eval_metric="mlogloss",
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
    num_class=4,
)

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ],
)

pipeline.fit(X_train, y_train)  # , estimator__sample_weight=w_train)
y_proba = pipeline.predict_proba(X_test)
y_proba[:5]

# %% [markdown]
# # Evaluation

# %%
# TODO: replace metrics with multiclass metrics
# y_pred = y_proba > 0.5
# for idx, target in enumerate(target_cols):
#     print(f"{target}")
#     print("-" * len(target))
#     print(f"accuracy: {accuracy_score(y_test[target], y_pred[:, idx])}")
#     print(f"confusion matrix:\n{confusion_matrix(y_test[target], y_pred[:, idx])}")
#     print()

# %% [markdown]
# # Ranking hires based on predictions and heuristic

# %%
# This heuristic is based on the actual scoring mechanism
# TODO: proba may need to be converted from logistic to some other distribution?
hr_scores = np.zeros((y_proba.shape[0]))
hr_scores = (
    hr_scores[:] + 0.25 * y_proba[:, 1] + 0.25 * y_proba[:, 2] + 0.5 * y_proba[:, 3]
)
hr_score_col = "HR_SCORE"
targets_hired = targets_test.copy()
targets_hired[hr_score_col] = hr_scores
targets_hired.sort_values(by=[hr_score_col], ascending=False, inplace=True)
targets_hired = targets_hired.head(targets_hired.shape[0] // 2)
targets_hired.head()

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
) = extract_counts(targets_hired)

(
    thp_count_test,
    tr_count_test,
    thpr_count_test,
    pro_count_test,
    priv_count_test,
) = extract_counts(targets_test)

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
