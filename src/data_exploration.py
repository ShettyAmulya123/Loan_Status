import pandas as pd

df = pd.read_excel("data/loan_approval_dataset.xlsx")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.columns = df.columns.str.strip()
print(df["loan_status"].value_counts())