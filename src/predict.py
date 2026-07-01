import pickle

# Load trained model
with open("models/loan_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")

"""
# Sample customer data
customer = [
    [
        2,  # no_of_dependents
        0,  # education (Graduate)
        0,  # self_employed (No)
        800000,  # income_annum
        3000000,  # loan_amount
        15,  # loan_term
        780,  # cibil_score
        5000000,  # residential_assets_value
        2000000,  # commercial_assets_value
        1000000,  # luxury_assets_value
        3000000,  # bank_asset_value
    ]
]
"""

import pandas as pd

customer = pd.DataFrame(
    [
        {
            "no_of_dependents": 2,
            "education": 0,
            "self_employed": 0,
            "income_annum": 800000,
            "loan_amount": 3000000,
            "loan_term": 15,
            "cibil_score": 780,
            "residential_assets_value": 5000000,
            "commercial_assets_value": 2000000,
            "luxury_assets_value": 1000000,
            "bank_asset_value": 3000000,
        }
    ]
)

prediction = model.predict(customer)
print(f"prediction: {prediction}")

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")

probability = model.predict_proba(customer)
print(f"probability: {probability}")
