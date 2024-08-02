import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()  # Drop rows with missing values
    return df


def preprocess_data(df):
    if 'LeadStatus' not in df.columns:
        raise KeyError("Column 'LeadStatus' not found in DataFrame")

    # Convert LeadStatus to binary target variable
    df['interested'] = df['LeadStatus'].apply(lambda x: 1 if x in ['Warm', 'Hot'] else 0)

    # Drop the original LeadStatus column
    X = df.drop(['LeadStatus', 'interested'], axis=1)
    y = df['interested']

    # Check if there are at least two classes
    if len(set(y)) < 2:
        raise ValueError(
            "The data contains only one class. Please ensure the dataset contains both 'Cold' and 'Warm'/'Hot' entries.")

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Save column names before scaling
    columns = X.columns

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, columns


def train_model(X, y, columns, scaler):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)  # Increase max_iter
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model accuracy: {accuracy}')

    # Save the model to disk
    joblib.dump(model, 'model.joblib')
    joblib.dump(columns, 'model_columns.joblib')
    joblib.dump(scaler, 'scaler.joblib')


file_path = '/Users/bhuvaneshvarnarayan/Documents/GitHub/referral_system/data/customer_conversion_traing_dataset.csv'
df = load_data(file_path)
try:
    X, y, scaler, columns = preprocess_data(df)
    train_model(X, y, columns, scaler)
except ValueError as e:
    print(e)
