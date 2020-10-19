import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from preprocessing import get_data


def knn_model():
    X_train, X_test, y_train, y_test = get_data('Admission_Predict.csv', False)
    knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


def main():
    acc = []
    for i in range(20):
        acc.append(knn_model())
    print("Average accuracy using KNN: {}".format(np.mean(acc)))


if __name__ == '__main__':
    main()