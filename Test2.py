import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Loading the admission dataset from the csv file using pandas
datafile = 'Admission_Predict.csv'
df = pd.read_csv(datafile, header=0)
colnames  = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
reorder_colnames = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'RACE', 'SES', 'ADM']
features = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'RACE', 'ADM']
df.columns = colnames
df.reindex(columns=reorder_colnames)
numerical = ['GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES']
print("Number of rows: " + str(len(df.index)))
df.dropna(inplace=True)
print("Number of valid rows without missing values: " + str(len(df.index)))
# print(data.head(10))
y = df['ADM']
X = df[numerical]
std = StandardScaler()
std.fit(X)

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Slope: %.3f' % lr.coef_[0])
print('Intercept: %.3f' % lr.intercept_)
print('R2 score: %.3f' % r2_score(y_test, y_pred_lr))
print('Mean squared error: %.3f' % mean_squared_error(y_test, y_pred_lr))