import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()  # Drop rows with missing values
    return df

def preprocess_data(df):
    if 'LeadStatus' not in df.columns:
        raise KeyError("Column 'LeadStatus' not found in DataFrame")

    df['interested'] = df['LeadStatus'].apply(lambda x: 1 if x in ['Warm', 'Hot'] else 0)

    features = [
        'Age', 'Gender', 'Location', 'LeadSource', 'TimeSpent (minutes)',
        'PagesViewed', 'EmailSent', 'DeviceType',
        'FormSubmissions', 'CTR_ProductPage',
        'ResponseTime (hours)', 'FollowUpEmails',
        'SocialMediaEngagement', 'PaymentHistory'
    ]

    X = df[features]
    y = df['interested']

    if len(set(y)) < 2:
        raise ValueError("The data contains only one class. Please ensure the dataset contains both 'Cold' and 'Warm'/'Hot' entries.")

    X = pd.get_dummies(X, drop_first=True)
    columns = X.columns

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    return X_scaled, y, scaler, poly, columns

def train_model(X, y, scaler, poly, columns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid_log_reg = {'max_iter': [500, 1000], 'solver': ['lbfgs', 'liblinear']}
    log_reg = LogisticRegression()
    log_reg_cv = GridSearchCV(log_reg, param_grid_log_reg, cv=5)
    log_reg_cv.fit(X_train, y_train)
    best_log_reg = log_reg_cv.best_estimator_
    print(f'Best parameters for log_reg: {log_reg_cv.best_params_}')

    param_grid_rf = {'n_estimators': [100, 200], 'max_features': ['sqrt']}
    rf = RandomForestClassifier()
    rf_cv = GridSearchCV(rf, param_grid_rf, cv=5)
    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    print(f'Best parameters for rf: {rf_cv.best_params_}')

    param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
    gb = GradientBoostingClassifier()
    gb_cv = GridSearchCV(gb, param_grid_gb, cv=5)
    gb_cv.fit(X_train, y_train)
    best_gb = gb_cv.best_estimator_
    print(f'Best parameters for gb: {gb_cv.best_params_}')

    param_grid_xgb = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_cv = GridSearchCV(xgb, param_grid_xgb, cv=5)
    xgb_cv.fit(X_train, y_train)
    best_xgb = xgb_cv.best_estimator_
    print(f'Best parameters for xgb: {xgb_cv.best_params_}')

    param_grid_lgbm = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}
    lgbm = LGBMClassifier()
    lgbm_cv = GridSearchCV(lgbm, param_grid_lgbm, cv=5)
    lgbm_cv.fit(X_train, y_train)
    best_lgbm = lgbm_cv.best_estimator_
    print(f'Best parameters for lgbm: {lgbm_cv.best_params_}')

    models = [best_log_reg, best_rf, best_gb, best_xgb, best_lgbm]
    model_names = ['log_reg', 'rf', 'gb', 'xgb', 'lgbm']

    for model, name in zip(models, model_names):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        print(f'Accuracy for {name}: {accuracy}')
        print(f'Classification report for {name}:\n{report}')

    ensemble = VotingClassifier(estimators=[(name, model) for name, model in zip(model_names, models)], voting='hard')
    ensemble.fit(X_train, y_train)
    ensemble_predictions = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    ensemble_report = classification_report(y_test, ensemble_predictions)
    print(f'Ensemble model accuracy: {ensemble_accuracy}')
    print(f'Ensemble model classification report:\n{ensemble_report}')

    joblib.dump(ensemble, 'model.joblib')
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
