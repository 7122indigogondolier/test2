import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data(datafile, continuous=False):
    # Loading the admission dataset from the csv file using pandas
    df = pd.read_csv(datafile, header=0)
    colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
    reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']
    df.columns = colnames

    # Dropping rows with at least one missing value
    # print("Number of rows: " + str(len(df.index)))
    df.dropna(inplace=True)
    # print("Number of valid rows without missing values: " + str(len(df.index)))

    # Reordering the columns and encoding the 'RACE' column with onehot encoding
    df = df.reindex(columns=reorder_colnames)
    df = pd.get_dummies(df, columns=['RACE'])

    if not continuous:
        # Transforming the output into binary classes using threshold of 0.73
        df.loc[(df.ADM >= 0.73), 'ADM'] = 1
        df.loc[(df.ADM < 0.73), 'ADM'] = 0

    # Defining features and target
    features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES',
                'RACE_Asian', 'RACE_african american', 'RACE_latinx', 'RACE_white']
    remove_SES = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES',
                'RACE_Asian', 'RACE_african american', 'RACE_latinx', 'RACE_white']
    X = df[features]
    # X = df[remove_SES]
    y = df['ADM']

    # Scaling the features
    std = StandardScaler()
    std.fit(X)

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


