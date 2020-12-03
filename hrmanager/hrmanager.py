# %%
import pandas as pd
from schema import INDEX_COL, PREDICTOR_COLS, PROTECTED_GROUP_COL, RETAINED_COL
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# %%
usecols = [INDEX_COL, PROTECTED_GROUP_COL, RETAINED_COL] + list(PREDICTOR_COLS)
train = pd.read_csv(
    "../data/train.csv",
    index_col="UNIQUE_ID",
    usecols=usecols,
    na_values=" "
)
train.head()

# %%
X = train[list(PREDICTOR_COLS)]
y = train[RETAINED_COL]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', XGBClassifier())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# %%
accuracy_score(y_test, y_pred)
