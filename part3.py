import numpy as np
import pandas as pd

df = pd.read_csv('student_scores.csv')

df['math_score'] = df['math_score'].fillna(df['math_score'].mean())

df['science_score'] = df['science_score'].fillna(df['science_score'].mean())

df['exam_date'] = pd.to_datetime(
    df['exam_date'],
    dayfirst=True,
    errors='coerce'
)

Q1 = df['math_score'].quantile(0.25)
Q3 = df['math_score'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df['math_score'] >= lower) &
    (df['math_score'] <= upper)
]

df = df.drop_duplicates()

print(df)