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

datafile = 'Admission_Predict.csv'
df = pd.read_csv(datafile, header=0)
colnames  = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES','RACE', 'SES', 'ADM']
features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ADM']
# serial number is omitted from the features because serial number should not matter in predicting
# admissions
# I have reordered the columns so that the column with the target value (admission likelihood) is
# the last column (makes our graphs easier to read)


df.columns = colnames
df.reindex(columns=reorder_colnames)

# We need to find a way to convert race into numerical values
# We need to fill out/remove the NaN and Null values
    # For this we can either fill in the values or,
    # we can remove rows containing null / NaN values
# Once we finish these things, we will finally be able to plot the scatterplot and correlation matrix



print(df.head())
scatterplotmatrix(df[features].values, alpha=0.5, figsize=(10,8), names=features)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[features].values.T)
hm = heatmap(cm, row_names=features, column_names=features)
plt.show()
