import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib

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

def train_model(X, y, scaler, poly, columns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    lgbm = LGBMClassifier(random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'log_reg': {
            'max_iter': [500, 1000],
            'solver': ['lbfgs', 'liblinear']
        },
        'rf': {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 'log2']  # Fixed to valid options
        },
        'gb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        },
        'lgbm': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }
    }

    models = {
        'log_reg': log_reg,
        'rf': rf,
        'gb': gb,
        'xgb': xgb,
        'lgbm': lgbm
    }

    best_estimators = {}
    for model_name in models.keys():
        grid_search = GridSearchCV(models[model_name], param_grid[model_name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Ensemble model
    voting_clf = VotingClassifier(estimators=[
        ('log_reg', best_estimators['log_reg']),
        ('rf', best_estimators['rf']),
        ('gb', best_estimators['gb']),
        ('xgb', best_estimators['xgb']),
        ('lgbm', best_estimators['lgbm'])
    ], voting='soft')
    voting_clf.fit(X_train, y_train)

    # Evaluate models
    for model_name, model in best_estimators.items():
        y_pred = model.predict(X_test)
        print(f"\nAccuracy for {model_name}: {accuracy_score(y_test, y_pred)}")
        print(f"Classification report for {model_name}:\n{classification_report(y_test, y_pred)}")

    # Evaluate ensemble model
    ensemble_pred = voting_clf.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"\nEnsemble model accuracy: {ensemble_accuracy}")
    print(f"Ensemble model classification report:\n{classification_report(y_test, ensemble_pred)}")

    # Save the best ensemble model to disk
    joblib.dump(voting_clf, 'model.joblib')
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
