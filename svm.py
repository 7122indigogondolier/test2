"""
Team: Utkrist P. Thapa '21
      Abhi Jha '21
      Tina Jin '21
svm.py: Implementing SVM to predict admission
"""

from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    # retrieve the data from the file
    datafile = "Admission_Predict.csv"
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

    # initializing and fitting the svm to our training data
    svm = SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=1)
    svm.fit(x_train_std, y_train)

    # using the svm to mke predictions on test set
    y_pred = svm.predict(x_test_std)

    # display the performance of the model using some evaluation metrics
    print("")
    print("-----Testing Performance of the SVM on unseen test data-----")
    print("")
    print("F-1 score: %f, Precision score: %f, Recall score: %f" %
    (f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("")
    print("-----")
    print("Utkrist P. Thapa '21")
    print("Abhi Jha '21")
    print("Tina Jin '21")
    print("Washington and Lee University")
    print("")

if __name__=='__main__':
    main()
