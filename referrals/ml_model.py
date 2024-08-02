import pandas as pd
import joblib
import os

# Load the model, columns, and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
columns_path = os.path.join(os.path.dirname(__file__), 'model_columns.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)
scaler = joblib.load(scaler_path)

def predict(data):
    # Match the structure of the training data
    X = pd.DataFrame([data])
    X = pd.get_dummies(X, drop_first=True)

    # Ensure the test data has the same columns as the training data
    missing_cols = set(model_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0

    # Reorder columns to match training data
    X = X[model_columns]

    # Scale the features
    X = scaler.transform(X)

    prediction = model.predict(X)
    return int(prediction[0])
