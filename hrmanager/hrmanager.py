# %%
import pandas as pd
from schema import HIGH_PERFORMER_COL, INDEX_COL, PREDICTOR_COLS, \
    PROTECTED_GROUP_COL, RETAINED_COL
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# %%
predictor_cols = list(PREDICTOR_COLS)
target_cols = [HIGH_PERFORMER_COL, PROTECTED_GROUP_COL, RETAINED_COL]
usecols = [INDEX_COL] + target_cols + predictor_cols
train = pd.read_csv(
    "../data/train.csv",
    index_col="UNIQUE_ID",
    usecols=usecols,
    na_values=" "
)
train.dropna(subset=target_cols, inplace=True)

X = train[predictor_cols]
y = train[target_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)

# %%
estimator = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model = OneVsRestClassifier(estimator)

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('model', model)
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict_proba(X_test)

# %%
y_thr = y_pred > 0.5
print(accuracy_score(y_test[[HIGH_PERFORMER_COL]], y_thr[:, 0]))
print(accuracy_score(y_test[[PROTECTED_GROUP_COL]], y_thr[:, 1]))
print(accuracy_score(y_test[[RETAINED_COL]], y_thr[:, 2]))

# %%
