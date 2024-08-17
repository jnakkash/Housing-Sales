import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_log_error
from scipy.stats import randint

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

# Initialize models
lasso = Lasso(alpha=0.001, random_state=42)
ridge = Ridge(alpha=0.001, random_state=42)
elasticnet = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

# Train individual models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
elasticnet.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# Evaluate individual models
models = [('Lasso', lasso), ('Ridge', ridge), ('ElasticNet', elasticnet), ('RandomForest', rf), ('GradientBoosting', gbr)]
for name, model in models:
    val_preds = model.predict(X_val)
    val_score = np.sqrt(mean_squared_log_error(y_val, val_preds))
    print(f"{name} Validation RMSLE: {val_score:.4f}")

# Ensemble with stacking
stacked_model = StackingRegressor(
    estimators=models,
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)

stacked_model.fit(X_train, y_train)

# Evaluate the stacked model
val_preds = stacked_model.predict(X_val)
val_score = np.sqrt(mean_squared_log_error(y_val, val_preds))
print(f"Stacked Model Validation RMSLE: {val_score:.4f}")

# Generate predictions for the test set
test_preds = np.expm1(stacked_model.predict(test_preprocessed))  # Reverse log1p transformation

# Prepare submission
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_preds
})

submission.to_csv('/workspaces/Housing-Sales/submission.csv', index=False)
print(submission.head())