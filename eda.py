"""
Team: Utkrist P. Thapa '21
      Abhi Jha '21
      Tina Jin '21
eda.py: This program contains exploratory data analysis of the Admission_Predict.csv data file
"""
import pandas as pd
import numpy as np
from mlxtend.plotting import scatterplotmatrix, heatmap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

datafile = 'Admission_Predict.csv'
df = pd.read_csv(datafile, header=0)
colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']

df.columns = colnames
df.dropna(inplace=True)
df = df.reindex(columns=reorder_colnames)
df = pd.get_dummies(df, columns=['RACE'])
outputlast = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
       'RACE_white', 'ADM']
df = df.reindex(columns=outputlast)
features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'RACE_Asian', 'RACE_african american', 'RACE_latinx',
       'RACE_white', 'ADM']

print(df['ADM'].describe())
plt.hist(df['ADM'])
plt.show()
scatterplotmatrix(df[features].values, alpha=0.5, figsize=(10,8), names=features)
# plt.tight_layout()
plt.show()

cm = np.corrcoef(df[features].values.T)
hm = heatmap(cm, row_names=features, column_names=features)
plt.show()
