import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load data
train_data = pd.read_csv('/workspaces/Housing-Sales/train.csv')
test_data = pd.read_csv('/workspaces/Housing-Sales/test.csv')

# Define features and target
X = train_data.drop(columns=['SalePrice'])
y = np.log1p(train_data['SalePrice'])  # Apply log transformation to SalePrice

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Define the transformers for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply the preprocessor to the dataset
X_preprocessed = preprocessor.fit_transform(X)
X_test_preprocessed = preprocessor.transform(test_data)

# Split the preprocessed data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV for the best model
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Best Model RMSE: {rmse}')

# Final prediction on the test data
final_predictions = np.expm1(best_model.predict(X_test_preprocessed))

# Prepare submission file
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': final_predictions})
submission.to_csv('/workspaces/Housing-Sales/submission.csv', index=False)
print("Submission file created.")