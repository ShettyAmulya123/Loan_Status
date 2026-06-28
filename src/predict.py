import pickle

model = pickle.load(
    open("models/loan_model.pkl", "rb")
)

print("Model loaded successfully!")
