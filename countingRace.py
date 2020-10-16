"""
Team: Utkrist P. Thapa '21
      Abhi Jha '21
      Tina Jin '21
countingRace.py: This progra counts the number of each race in the dataset and outputs the info in a
file
"""

import pandas as pd

def main():
    filename = 'Admission_Predict.csv'
    df = pd.read_csv(filename, header=0)
    colnames  = ['SN', 'GRE', 'TOEFL', 'URATE', 'SOP', 'LOR', 'CGPA', 'RES', 'ADM', 'RACE', 'SES']
    df.columns = colnames

    record = dict()

    for item in df['RACE']:
        record[item] = 0

    for item in df['RACE']:
        record[item] += 1

    filewriter = open('race_info.txt', 'w')
    for key in record:
        filewriter.write("%20s: %6d \n" % (key, record[key]))
    filewriter.close()


if __name__=='__main__':
    main()
