# %%
import autosklearn.classification
import pandas as pd
from preprocess import predictors_dataset
from schema import BINARY_PREDICTOR_COLS, NUMERIC_PREDICTOR_COLS, RETAINED_COL
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %%
df = predictors_dataset("../data/train.csv", target_cols=[RETAINED_COL])
df.head()

# %%
X = df[list(BINARY_PREDICTOR_COLS + NUMERIC_PREDICTOR_COLS)]

# NOTE: set types so that automl can infer them correctly
for col in BINARY_PREDICTOR_COLS:
    X[col] = X[col].astype("category")
    if X[col].isnull().values.any():
        print(f"{col} contains nan")

for col in NUMERIC_PREDICTOR_COLS:
    X[col] = pd.to_numeric(X[col])
    if X[col].isnull().values.any():
        print(f"{col} contains nan")

y = pd.DataFrame(df[RETAINED_COL], dtype='category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=3
)

# %%
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train, X_test, y_test)

# %%
predictions = automl.predict(X_test)
print("Accuracy score:", accuracy_score(y_test, predictions))

# %%
