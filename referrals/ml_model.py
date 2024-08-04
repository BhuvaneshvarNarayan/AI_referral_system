import pandas as pd
import joblib
import os

# Load the model, columns, and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
poly_path = os.path.join(os.path.dirname(__file__), 'poly.joblib')
columns_path = os.path.join(os.path.dirname(__file__), 'model_columns.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
poly = joblib.load(poly_path)
trained_columns = joblib.load(columns_path)

def predict(data):
    # Match the structure of the training data
    X = pd.DataFrame([data])
    X = pd.get_dummies(X, drop_first=True)

    # Add missing columns with default value of 0
    for col in trained_columns:
        if col not in X.columns:
            X[col] = 0

    # Reorder columns to match training data
    X = X[trained_columns]

    # Polynomial Features
    X_poly = poly.transform(X)

    # Scale the features
    X_scaled = scaler.transform(X_poly)

    prediction = model.predict(X_scaled)
    return int(prediction[0])