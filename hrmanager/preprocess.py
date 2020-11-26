# %%
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../data/train.csv", index_col="UNIQUE_ID")
criterion_df = df.loc[:, "Overall_Rating":"Retained"]
predictors_df = df.loc[:, "SJ_Most_1":"PScale13_Q5"]

# %%
numeric_cols = ["Scenario1_Time", "Scenario2_Time"] \
    + [f"SJ_Time_{number}" for number in range(1, 10)]
binary_features_df = predictors_df.drop(numeric_cols, axis=1)
numeric_features_df = predictors_df.loc[:, numeric_cols]

# %%
X = OneHotEncoder().fit_transform(binary_features_df)
Y = criterion_df["Retained"].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# %%
estimator = AdaBoostClassifier()
estimator.fit(X_train, Y_train)
Y_pred = estimator.predict(X_test)

# %%
confusion_matrix(Y_test, Y_pred)

# %%
accuracy_score(Y_test, Y_pred)
# %%
