import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_excel("data/loan_approval_dataset.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Remove leading/trailing spaces from values
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

printdf["education"]


# Encode values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["education"] = le.fit_transform(df["education"])

# mannually encode self_employed and loan_status
df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions[:10])
print("Model trained successfully!")