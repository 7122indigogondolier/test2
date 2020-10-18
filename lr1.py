"""
Team: Utkrist P. Thapa '21
      Abhi Jha '21
      Tina Jin '21
logsitic_regression.py: This program contains our implementation of logistic regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def preprocess(datafile):
    df = pd.read_csv(datafile, header=0)
    colnames  = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP',
        'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
    reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP',
        'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']

    df.columns = colnames
    df.dropna(inplace=True)
    df = df.reindex(columns=reorder_colnames)

    df = pd.get_dummies(df, columns=['RACE'])
    outputlast = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP',
        'LOR', 'CGPA', 'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
        'RACE_white', 'ADM']
    df = df.reindex(columns=outputlast)
    return df

def main():
    datafile = 'Admission_Predict.csv'
    df = preprocess(datafile)
    features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA',
        'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
        'RACE_white']
    target = 'ADM'

    x = df[features]
    source_y = []
    for item in df[target]:
        if item < 0.73:
            source_y.append(0)
        else:
            source_y.append(1)
    y = pd.DataFrame(source_y, columns=[target])
    y = y[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    log_reg = LogisticRegression(solver='lbfgs')
    log_reg.fit(x_train_std, y_train)
    predictions = log_reg.predict(x_test_std)
    print("Accuracy: %.3f" % (accuracy_score(y_test, predictions)))

if __name__ == '__main__':
    main()
