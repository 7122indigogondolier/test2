import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Loading the admission dataset from the csv file using pandas
datafile = 'Admission_Predict.csv'
df = pd.read_csv(datafile, header=0)
colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ADM', 'RACE']
df.columns = colnames

# Dropping rows with at least one missing value
print("Number of rows: " + str(len(df.index)))
df.dropna(inplace=True)
print("Number of valid rows without missing values: " + str(len(df.index)))

# Reordering the columns and encoding the 'RACE' column with onehot encoding
df = df.reindex(columns=reorder_colnames)
df = pd.get_dummies(df, columns=['RACE'])
features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES',
            'RACE_Asian', 'RACE_african american', 'RACE_latinx', 'RACE_white']

# Transforming the output into binary classes using threshold of 0.73
df[df['ADM'] >= 0.73] = 1
df[df['ADM'] < 0.73] = 0

# Defining features and output
y = df['ADM']
X = df[features]

# Scaling the features
std = StandardScaler()
std.fit(X)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


