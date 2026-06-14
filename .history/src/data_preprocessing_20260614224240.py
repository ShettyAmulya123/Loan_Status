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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)