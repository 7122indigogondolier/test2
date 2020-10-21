"""
Team: Utkrist P. Thapa '21
      Abhi Jha '21
      Tina Jin '21
eda2.py: This program contains exploratory data analysis of the Admission_Predict.csv data file
"""

import pandas as pd
import matplotlib.pyplot as plt

def plotHist(df, features):
    [f1, f2, f3, f4] = features
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Distribution of features')
    axes[0, 0].hist(df[f1])
    axes[0, 0].set(xlabel=f1, ylabel='Frequency')
    axes[0, 1].hist(df[f2])
    axes[0, 1].set(xlabel=f2, ylabel='Frequency')
    axes[1, 0].hist(df[f3])
    axes[1, 0].set(xlabel=f3, ylabel='Frequency')
    axes[1, 1].hist(df[f4])
    axes[1, 1].set(xlabel=f4, ylabel='Frequency')
    plt.show()

def main():
    datafile = 'Admission_Predict.csv'
    df = pd.read_csv(datafile, header=0)
    colnames  = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
    reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']

    df.columns = colnames
    df.dropna(inplace=True)
    df = df.reindex(columns=reorder_colnames)

    df = pd.get_dummies(df, columns=['RACE'])
    outputlast = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
           'RACE_white', 'ADM']
    df = df.reindex(columns=outputlast)
    features = ['LOR', 'CGPA', 'RES', 'SES']
    plotHist(df, features)

if __name__ == '__main__':
    main()
