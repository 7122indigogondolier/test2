import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from preprocessing import get_data

X_train, X_test, y_train, y_test, features = get_data('Admission_Predict.csv', False)
dt = DecisionTreeClassifier(criterion='gini', max_depth=10)
dt.fit(X_train, y_train)
tree_pred = dt.predict(X_test)
print(dt.feature_importances_)
print("Decision Tree Accuracy: %3f" % accuracy_score(y_test, tree_pred))