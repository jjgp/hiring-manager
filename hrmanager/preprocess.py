# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# %%
df = pd.read_csv("../data/train.csv", index_col="UNIQUE_ID", na_values=" ")
criterion_df = df.loc[:, "Overall_Rating":"Retained"]
predictors_df = df.loc[:, "SJ_Most_1":"PScale13_Q5"]

# %%
numeric_cols = ["Scenario1_Time", "Scenario2_Time"] \
    + [f"SJ_Time_{number}" for number in range(1, 10)]
binary_features_df = predictors_df.drop(numeric_cols, axis=1)
binary_features_df.fillna(0, inplace=True)
numeric_features_df = predictors_df.loc[:, numeric_cols]
for col in numeric_cols:
    mode = numeric_features_df[col].mode(dropna=True).iloc[-1]
    numeric_features_df[col] = numeric_features_df[col].fillna(mode)

# %%
X = OneHotEncoder().fit_transform(binary_features_df).toarray()
X = np.concatenate((X, numeric_features_df.to_numpy()), axis=1)
Y = criterion_df["Retained"].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# %%
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

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# %%
