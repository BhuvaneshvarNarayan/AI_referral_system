import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import joblib
import os

# Define the path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')
POLY_PATH = os.path.join(os.path.dirname(__file__), 'poly.joblib')
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), 'model_columns.joblib')

# Load the trained models and preprocessing objects
best_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
poly = joblib.load(POLY_PATH)
model_columns = joblib.load(COLUMNS_PATH)


def preprocess_input(data):
    # Convert input data to DataFrame
    X = pd.DataFrame([data])

    # Handle missing columns by adding them with default value 0
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0

    # Ensure the same order of columns as in training
    X = X[model_columns]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Convert the imputed array back to DataFrame
    X_imputed_df = pd.DataFrame(X_imputed, columns=model_columns)

    # Encode categorical variables (if any)
    X_encoded = pd.get_dummies(X_imputed_df, columns=X_imputed_df.select_dtypes(include=['object']).columns,
                               drop_first=True)

    # Ensure the same order of columns as in training
    missing_cols = set(model_columns) - set(X_encoded.columns)
    for col in missing_cols:
        X_encoded[col] = 0
    X_encoded = X_encoded[model_columns]

    # Polynomial Features
    X_poly = poly.transform(X_encoded)

    # Scale the features
    X_scaled = scaler.transform(X_poly)

    return X_scaled


def predict(data):
    # Preprocess input data
    X_preprocessed = preprocess_input(data)

    # Predict using the best model
    prediction = best_model.predict(X_preprocessed)
    return int(prediction[0])