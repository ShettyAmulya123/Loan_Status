import pandas as pd

df = pd.read_excel("data/loan_approval_dataset.xlsx")

df.columns = df.columns.str.strip()

df = df.drop("loan_id", axis=1)

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)
print(y.head())

print(df["education"].unique())

print(df["self_employed"].unique())

print(df["loan_status"].unique())

# Remove spaces from categorical values

df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["education"] = le.fit_transform(df["education"])


# encoding using the manual method
df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

print(df.head())

print(df["education"].unique())
print(df["self_employed"].unique())
print(df["loan_status"].unique())

Deviation Dataset
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Evaluation