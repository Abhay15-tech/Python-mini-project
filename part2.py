import numpy as np
import pandas as pd
df= pd.read_csv('student_scores.csv') 
print("\nfirst 5 rows of the dataset:",df.head())
print("\n  data types of all columns",df.dtypes)
print("\n  total number of missing values in each column",df.isnull().sum())
print("\n Attendanace:", df[df['attendance'] < 70])