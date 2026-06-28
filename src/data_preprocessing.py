import pandas as pd

# ===============================
# Load Dataset
# ===============================
df = pd.read_excel("data/loan_approval_dataset.xlsx")

# ===============================
# Clean Column Names
# ===============================
df.columns = df.columns.str.strip()

# ===============================
# Remove Extra Spaces from Values
# ===============================
df["education"] = df["education"].str.strip()
df["self_employed"] = df["self_employed"].str.strip()
df["loan_status"] = df["loan_status"].str.strip()

# ===============================
# Encode Categorical Columns
# ===============================
from sklearn.preprocessing import LabelEncoder

education_encoder = LabelEncoder()

df["education"] = education_encoder.fit_transform(
    df["education"]
)

# Manual Encoding
df["self_employed"] = df["self_employed"].map({
    "No": 0,
    "Yes": 1
})

df["loan_status"] = df["loan_status"].map({
    "Rejected": 0,
    "Approved": 1
})

# ===============================
# Remove Unnecessary Column
# ===============================
df.drop("loan_id", axis=1, inplace=True)

# ===============================
# Features and Target
# ===============================
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ===============================
# Train-Test Split
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Train Random Forest Model
# ===============================
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# ===============================
# Predictions
# ===============================
predictions = model.predict(X_test)

print("Sample Predictions:")
print(predictions[:10])

# ===============================
# Model Evaluation
# ===============================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nModel Performance")
print("-------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ===============================
# Confusion Matrix
# ===============================
print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# ===============================
# Classification Report
# ===============================
print("\nClassification Report")
print(classification_report(y_test, predictions))

# ===============================
# Feature Importance
# ===============================
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(feature_importance)

