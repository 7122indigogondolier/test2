import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from preprocessing import get_data


def linear_model():
    X_train, X_test, y_train, y_test = get_data('Admission_Predict.csv', True)
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_test)
    return r2_score(y_test, y_pred)


def main():
    acc = []
    for i in range(20):
        acc.append(linear_model())
    print(np.mean(acc))


if __name__ == '__main__':
    main()