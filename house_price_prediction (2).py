import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preparation
# Load the training and test data
train_df = pd.read_csv('/workspaces/Housing-Sales/train.csv')
test_df = pd.read_csv('/workspaces/Housing-Sales/test.csv')

# Print column names for debugging
print("Training data columns:", train_df.columns)
print("Test data columns:", test_df.columns)

# Ensure the 'Id' column exists; if not, skip that step
if 'Id' in train_df.columns:
    X = train_df.drop(columns=['SalePrice', 'Id'])
    test_ids = test_df['Id']
else:
    X = train_df.drop(columns=['SalePrice'])
    test_ids = None

y = train_df['SalePrice']

# Data Cleaning
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply transformations to training data
X = preprocessor.fit_transform(X)

# Apply the same transformations to test data, if 'Id' exists
if test_ids is not None:
    X_test = preprocessor.transform(test_df.drop(columns=['Id']))
else:
    X_test = preprocessor.transform(test_df)

# Step 2: Model Selection and Architecture Design
baseline_model = Ridge(alpha=1.0)
baseline_model.fit(X, y)
y_pred_baseline = baseline_model.predict(X)

# Evaluate baseline model
baseline_rmse = np.sqrt(mean_squared_error(y, y_pred_baseline))
baseline_r2 = r2_score(y, y_pred_baseline)
print(f"Baseline RMSE: {baseline_rmse}")
print(f"Baseline R²: {baseline_r2}")

# Advanced Model Selection: XGBoost and LightGBM
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

lgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'num_leaves': [20, 30, 40]
}

xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_grid, n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1)
lgb_random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=lgb_param_grid, n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1)

xgb_random_search.fit(X, y)
lgb_random_search.fit(X, y)

best_xgb = xgb_random_search.best_estimator_
best_lgb = lgb_random_search.best_estimator_

xgb_pred = best_xgb.predict(X)
lgb_pred = best_lgb.predict(X)

xgb_rmse = np.sqrt(mean_squared_error(y, xgb_pred))
lgb_rmse = np.sqrt(mean_squared_error(y, lgb_pred))

xgb_r2 = r2_score(y, xgb_pred)
lgb_r2 = r2_score(y, lgb_pred)

print(f"XGBoost RMSE: {xgb_rmse}")
print(f"XGBoost R²: {xgb_r2}")
print(f"LightGBM RMSE: {lgb_rmse}")
print(f"LightGBM R²: {lgb_r2}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_scores = cross_val_score(best_xgb, X, y, cv=kf, scoring='neg_mean_squared_error')
lgb_cv_scores = cross_val_score(best_lgb, X, y, cv=kf, scoring='neg_mean_squared_error')

print(f"XGBoost Cross-Validation RMSE: {np.sqrt(-xgb_cv_scores.mean())}")
print(f"LightGBM Cross-Validation RMSE: {np.sqrt(-lgb_cv_scores.mean())}")

ensemble_pred = (best_xgb.predict(X_test) + best_lgb.predict(X_test)) / 2

submission = pd.DataFrame({
    'Id': test_ids if test_ids is not None else np.arange(len(ensemble_pred)),
    'SalePrice': ensemble_pred
})
submission.to_csv('/workspaces/Housing-Sales/house_price_prediction.csv', index=False)

import joblib
joblib.dump(best_xgb, '/workspaces/Housing-Sales/best_xgb_model.pkl')
joblib.dump(best_lgb, '/workspaces/Housing-Sales/best_lgb_model.pkl')
