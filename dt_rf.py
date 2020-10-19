import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from preprocessing import get_data

def decision_tree_depth_check():
    X_train, X_test, y_train, y_test = get_data('Admission_Predict.csv', False)
    depth_check = [4, 8, 12, 16, 20] # 20 was optimal!
    for i in depth_check:
        tree_model = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
        dt = DecisionTreeClassifier(criterion='gini')
        dt.fit(X_train, y_train)
        tree_pred = dt.predict(X_test)
        print("Decision Tree Accuracy for depth {}: {}".format(i, accuracy_score(y_test, tree_pred)))

def random_forest_estimators_check():
    X_train, X_test, y_train, y_test = get_data('Admission_Predict.csv', False)
    estimators_check = [10, 25, 40, 55, 70, 85, 100, 200, 300] # 40 was optimal!
    for i in estimators_check:
        forest = RandomForestClassifier(criterion='gini', bootstrap=False, n_estimators=i, random_state=1, n_jobs=2)
        forest.fit(X_train, y_train)
        forest_pred = forest.predict(X_test)
        print("Random Forest Accuracy for {} estimators: {}".format(i, accuracy_score(y_test, forest_pred)))


def main():
    decision_tree_depth_check()
    random_forest_estimators_check()
    X_train, X_test, y_train, y_test = get_data('Admission_Predict.csv', False)
    forest = RandomForestClassifier(criterion='gini', bootstrap=False, n_estimators=40, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)
    print("Final accuracy using Random Forest: {}".format(accuracy_score(y_test, forest_pred)))


if __name__ == '__main__':
    main()