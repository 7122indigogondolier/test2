import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def get_data(datafile, continuous=False):
    # Loading the admission dataset from the csv file using pandas
    df = pd.read_csv(datafile, header=0)
    colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
    reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']
    df.columns = colnames

    # Dropping rows with at least one missing value
    df.dropna(inplace=True)

    # Reordering the columns and encoding the 'RACE' column with onehot encoding
    df = df.reindex(columns=reorder_colnames)
    df = pd.get_dummies(df, columns=['RACE'])

    if not continuous:
        # Transforming the output into binary classes using threshold of 0.73
        df.loc[(df.ADM >= 0.73), 'ADM'] = 1
        df.loc[(df.ADM < 0.73), 'ADM'] = 0

    # Defining features and target
    features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES',
                'RACE_Asian', 'RACE_african american', 'RACE_latinx', 'RACE_white']
    X = df[features]
    y = df['ADM']

    # Scaling the features
    std = StandardScaler()
    std.fit(X)

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def linear_regression():
    X_train, X_test, y_train, y_test = get_data('Admission_Predict.csv', True)
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_test)
    return y_test, y_pred, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)

def plot_pred(test, pred):
    plt.scatter(test, pred)
    plt.plot([0,1],[0,1])
    plt.xlabel('actual y')
    plt.ylabel('predicted y')
    plt.show()

def main():
    linear_out = linear_regression()
    print("R^2 score using Linear Regression: {}".format(linear_out[2]))
    print("Mean Squared Error using Linear Regression: {}".format(linear_out[3]))
    plot_pred(linear_out[0], linear_out[1])

if __name__ == '__main__':
    main()