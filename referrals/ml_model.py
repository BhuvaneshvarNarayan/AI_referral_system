import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE

from referrals.train_model import train_model

# Load the model and preprocessing objects
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
poly = joblib.load('poly.joblib')
model_columns = joblib.load('model_columns.joblib')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()  # Drop rows with missing values
    return df

def preprocess_data(df):
    if 'LeadStatus' not in df.columns:
        raise KeyError("Column 'LeadStatus' not found in DataFrame")

    # Convert LeadStatus to binary target variable
    df['interested'] = df['LeadStatus'].apply(lambda x: 1 if x in ['Warm', 'Hot'] else 0)

    # Select relevant features
    features = [
        'Age', 'Gender', 'Location', 'LeadSource', 'TimeSpent (minutes)',
        'PagesViewed', 'EmailSent', 'DeviceType',
        'FormSubmissions', 'CTR_ProductPage',
        'ResponseTime (hours)', 'FollowUpEmails',
        'SocialMediaEngagement', 'PaymentHistory'
    ]

    X = df[features]
    y = df['interested']

    # Check if there are at least two classes
    if len(set(y)) < 2:
        raise ValueError(
            "The data contains only one class. Please ensure the dataset contains both 'Cold' and 'Warm'/'Hot' entries.")

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Save column names before scaling
    columns = X.columns

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    return X_scaled, y, scaler, poly, columns

def preprocess_input(data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    print("Input DataFrame before processing:", input_df)

    # Ensure all required columns are present
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure the order of columns matches
    input_df = input_df[model_columns]
    print("Input DataFrame after matching columns:", input_df)

    # Polynomial Features
    X_poly = poly.transform(input_df)
    print("Polynomial features:", X_poly)

    # Scale the features
    X_scaled = scaler.transform(X_poly)
    print("Scaled features:", X_scaled)

    return X_scaled

def predict(data):
    X = preprocess_input(data)
    prediction = model.predict(X)
    print("Prediction:", prediction)
    return prediction[0]


file_path = '/Users/bhuvaneshvarnarayan/Documents/GitHub/referral_system/data/customer_conversion_training_dataset.csv'
df = load_data(file_path)
try:
    X, y, scaler, poly, columns = preprocess_data(df)
    train_model(X, y, scaler, poly, columns)
except ValueError as e:
    print(e)
