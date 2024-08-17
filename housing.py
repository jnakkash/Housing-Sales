import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_log_error

# Load data
train_df = pd.read_csv('/workspaces/Housing-Sales/train.csv')
test_df = pd.read_csv('/workspaces/Housing-Sales/test.csv')

# Split features and target
X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = np.log1p(train_df['SalePrice'])  # Apply log transformation to the target

# Preprocessing pipeline
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

# Apply transformations to both training and test data
X_preprocessed = preprocessor.fit_transform(X)
test_preprocessed = preprocessor.transform(test_df.drop(['Id'], axis=1))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Model initialization
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print(f"Cross-Validation RMSLE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_log_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation RMSLE: {np.sqrt(-grid_search.best_score_):.4f}")

# Initialize GradientBoostingRegressor for ensembling
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

# Ensemble with stacking
stacked_model = StackingRegressor(
    estimators=[('rf', best_rf), ('gbr', gbr)],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)

stacked_model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = stacked_model.predict(X_val)
val_score = np.sqrt(mean_squared_log_error(y_val, val_preds))
print(f"Validation RMSLE: {val_score:.4f}")

# Generate predictions for the test set
test_preds = np.expm1(stacked_model.predict(test_preprocessed))  # Reverse log1p transformation

# Prepare submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_preds
})

submission.to_csv('/workspaces/Housing-Sales/submission.csv', index=False)
print(submission.head())