# %%
import category_encoders as ce
import numpy as np
from IPython.display import display
from read_train import read_train
from schema import PREDICTOR_COLS
from schema import PROTECTED_GROUP_COL as TARGET
from schema import TARGET_COLS
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# %% [markdown]
# # Random seed

np.random.seed(1337)

# %% [markdown]
# # Extracting targets and predictors from train.csv

# %%
predictor_cols = list(PREDICTOR_COLS)
target_cols = list(TARGET_COLS)
train = read_train(target_cols)
train.dropna(subset=target_cols, inplace=True)
train.fillna(-1, inplace=True)

display(train)

X = train[predictor_cols]
y = train[TARGET]

# %% [markdown]
# # Train test split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
)

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

pipeline = Pipeline(
    steps=[
        ("counter", ce.CountEncoder(cols=predictor_cols)),
        ("estimator", estimator),
    ],
)
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)

# %% [markdown]
# # Evaluation

# %%
y_pred = np.argmax(y_proba, axis=1)
display(accuracy_score(y_test, y_pred))
display(confusion_matrix(y_test, y_pred))

# %%
