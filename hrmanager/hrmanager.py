# %%
from featurize import featurize
from preprocess import preprocess_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

df = preprocess_csv("../data/train.csv")
X, Y = featurize(df)
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
    [tree.feature_importances_ for tree in estimator.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# %%
