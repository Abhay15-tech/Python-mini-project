import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv("student_scores.csv")

# 2. Understanding the Dataset
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)

# 3. Checking Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Removing Duplicate Records
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)
df = df.drop_duplicates()

# 5. Handling Incorrect Data Types
if 'exam_date' in df.columns:
    df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')

# 6. Detecting and Handling Outliers (IQR Method)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# 7. Handling Categorical Variables
binary_cols = []

for col in categorical_cols:
    if df[col].nunique() == 2:
        binary_cols.append(col)

df = pd.get_dummies(df, columns=[col for col in categorical_cols if col not in binary_cols], drop_first=True)

for col in binary_cols:
    df[col] = df[col].astype('category').cat.codes

# 8. Removing Irrelevant Features
irrelevant_cols = ['student_id']

for col in irrelevant_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# 9. Handling Skewness
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    skewness = df[col].skew()

    if abs(skewness) > 1:
        df[col] = np.log1p(df[col])

# 10. Feature Scaling (StandardScaler)
scaler = StandardScaler()

numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save Cleaned Dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nFinal Shape:", df.shape)
print("Dataset preprocessing completed successfully.")
print("\nMissing Values:")
print(df.isnull().sum())