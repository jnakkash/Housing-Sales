import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

# Fix the import statement
from tensorflow.keras.layers import Dense, Dropout

# Load data
train_data = pd.read_csv('/workspaces/Housing-Sales/train.csv')
test_data = pd.read_csv('/workspaces/Housing-Sales/test.csv')

# Data Preprocessing
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])  # Apply log transformation to SalePrice

# Concatenate train and test data for consistent preprocessing
train_len = len(train_data)
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Feature Engineering
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']  # Create a total square footage feature

# Handle missing values for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Handle missing values for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Convert categorical features to numeric using Label Encoding
label_enc = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_enc.fit_transform(data[col].astype(str))

# Split the data back into train and test sets
train_data = data[:train_len]
test_data = data[train_len:]

# Separate features and target variable
X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']
X_test = test_data.drop(['SalePrice', 'Id'], axis=1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Check for NaNs in the test data before prediction
print("Checking for NaNs in the test data before prediction...")
print(np.isnan(X_test).sum())

# Identify the column causing NaN in the test data
nan_indices = np.where(np.isnan(X_test).sum(axis=1) > 0)[0]
if len(nan_indices) > 0:
    print(f"Found NaNs in the test data at indices: {nan_indices}")
    print("Corresponding rows in test data causing NaNs:")
    print(X_test[nan_indices])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1)

# Evaluate the model
val_loss, val_rmse = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation RMSE: {val_rmse}")

# Final prediction on the test data
final_predictions = np.expm1(model.predict(X_test).flatten())

# Check for NaN in predictions
nan_predictions_indices = np.where(np.isnan(final_predictions))[0]
if len(nan_predictions_indices) > 0:
    print(f"NaN prediction found at index: {nan_predictions_indices[0]}")
    print("Corresponding input features that led to NaN prediction:")
    print(X_test[nan_predictions_indices[0]])

# Prepare submission file
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': final_predictions})
submission.to_csv('/workspaces/Housing-Sales/submission_best.csv', index=False)
print("Submission file created.")

