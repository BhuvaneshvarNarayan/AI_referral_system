import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
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

    # Save column names before scaling
    columns = X.columns

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    return X_scaled, y, scaler, poly, columns


def train_model(X, y, scaler, poly, columns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression with Hyperparameter Tuning
    param_grid = {
        'max_iter': [500, 1000],
        'solver': ['lbfgs', 'liblinear']
    }
    log_reg = LogisticRegression()
    log_reg_cv = GridSearchCV(log_reg, param_grid, cv=5)
    log_reg_cv.fit(X_train, y_train)
    best_log_reg = log_reg_cv.best_estimator_

    # RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)

    # XGBClassifier
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    lgbm.fit(X_train, y_train)

    # Predictions
    log_reg_predictions = best_log_reg.predict(X_test)
    rf_predictions = rf.predict(X_test)
    gb_predictions = gb.predict(X_test)
    xgb_predictions = xgb.predict(X_test)
    lgbm_predictions = lgbm.predict(X_test)

    # Accuracies
    log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    gb_accuracy = accuracy_score(y_test, gb_predictions)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)

    print(f'Logistic Regression accuracy: {log_reg_accuracy}')
    print(f'Random Forest accuracy: {rf_accuracy}')
    print(f'Gradient Boosting accuracy: {gb_accuracy}')
    print(f'XGBoost accuracy: {xgb_accuracy}')
    print(f'LightGBM accuracy: {lgbm_accuracy}')

    # Choose the best model
    accuracies = {
        'log_reg': log_reg_accuracy,
        'rf': rf_accuracy,
        'gb': gb_accuracy,
        'xgb': xgb_accuracy,
        'lgbm': lgbm_accuracy
    }
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]

    best_model = {
        'log_reg': best_log_reg,
        'rf': rf,
        'gb': gb,
        'xgb': xgb,
        'lgbm': lgbm
    }[best_model_name]

    print(f'Best model ({best_model_name}) accuracy: {best_accuracy}')

    # Save the best model to disk
    joblib.dump(best_model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(poly, 'poly.joblib')
    joblib.dump(columns, 'model_columns.joblib')


file_path = '/Users/bhuvaneshvarnarayan/Documents/GitHub/referral_system/data/customer_conversion_training_dataset.csv'
df = load_data(file_path)
try:
    X, y, scaler, poly, columns = preprocess_data(df)
    train_model(X, y, scaler, poly, columns)
except ValueError as e:
    print(e)