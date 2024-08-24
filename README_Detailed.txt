### README File: Detailed Explanation of Each Step

This README file provides a step-by-step explanation of the Python script used to build a **House Prices Prediction Dashboard** using Dash. The script trains various machine learning models to predict house prices, compares their performance, and allows users to download the final predictions.

---

### Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Script Breakdown](#script-breakdown)
    1. [Importing Libraries](#importing-libraries)
    2. [Loading Data](#loading-data)
    3. [Data Preprocessing](#data-preprocessing)
    4. [Splitting Data](#splitting-data)
    5. [Defining Models](#defining-models)
    6. [Training and Evaluating Models](#training-and-evaluating-models)
    7. [Selecting the Best Model](#selecting-the-best-model)
    8. [Making Predictions](#making-predictions)
    9. [Creating the Dashboard Layout](#creating-the-dashboard-layout)
    10. [Dynamic Updates with Callbacks](#dynamic-updates-with-callbacks)
    11. [Download Feature](#download-feature)
5. [Running the App](#running-the-app)
6. [Conclusion](#conclusion)

---

### Introduction

This project builds an interactive dashboard that uses various machine learning models to predict house prices based on historical data. The dashboard allows users to:
- Compare different models.
- Visualize model performance.
- Download the final predictions in a CSV file.

### Installation

Before running the script, ensure you have the following Python libraries installed:
- Dash
- Pandas
- Plotly
- NumPy
- Scikit-learn

You can install these libraries using `pip`:

```bash
pip install dash pandas plotly numpy scikit-learn
```

### Data Preparation

The script expects two CSV files:
1. `train.csv` - Contains historical house data, including features and the sale price.
2. `test.csv` - Contains similar house features but without the sale price, which the models will predict.

Ensure these files are located in the specified paths within the script or update the paths accordingly.

### Script Breakdown

#### 1. Importing Libraries

**Purpose:** This step imports all the necessary libraries required to build the dashboard, process data, create models, and visualize results.

- **Dash**: For creating the web application and interactive components.
- **Pandas**: For reading, manipulating, and analyzing the datasets.
- **Plotly Express**: For generating interactive plots.
- **NumPy**: For numerical operations, such as linear regression for trendlines.
- **Scikit-learn**: For building and evaluating machine learning models.

```python
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

#### 2. Loading Data

**Purpose:** Load the training and test datasets into the script for further processing.

- **Training Data (`train.csv`)**: Contains the historical house prices and features that will be used to train the models.
- **Test Data (`test.csv`)**: Contains house features for which the script will predict prices.

```python
try:
    train_data = pd.read_csv('/workspaces/Housing-Sales/train.csv')
    test_data = pd.read_csv('/workspaces/Housing-Sales/test.csv')
except FileNotFoundError:
    print("Error: File not found. Please check the file paths.")
    exit()
```

#### 3. Data Preprocessing

**Purpose:** Prepare the data for model training by handling missing values, removing irrelevant columns, and converting categorical variables into numeric ones.

- **Filling Missing Values:** The `LotFrontage` column's missing values are filled with the median value.
- **Dropping Columns:** Irrelevant columns (e.g., `Alley`, `PoolQC`) that are either missing too much data or not useful for prediction are removed.
- **Encoding Categorical Data:** Text-based categorical variables are converted into numerical dummy variables.

```python
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())

train_data = train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, errors='ignore')
test_data = test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, errors='ignore')

train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)
```

#### 4. Splitting Data

**Purpose:** Split the data into a training set and a validation set.

- **Training Set:** Used to train the machine learning models.
- **Validation Set:** Used to evaluate the models' performance.

```python
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
test_data = test_data.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 5. Defining Models

**Purpose:** Define several machine learning models to be trained and compared.

- **Pipeline:** Each model is wrapped in a pipeline that handles preprocessing steps (like imputing missing data) and model fitting.
- **Models:** Various models, such as `RandomForest`, `GradientBoosting`, `LinearRegression`, etc., are defined.

```python
models = {
    'RandomForest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]),
    # ... Other models
}
```

#### 6. Training and Evaluating Models

**Purpose:** Train each model on the training data and evaluate its performance on the validation set.

- **RMSE (Root Mean Squared Error):** Measures the average error of the predictions. A lower RMSE indicates better performance.
- **R² (R-squared):** Indicates how well the model explains the variance in the data.

```python
model_performance = {}
y_pred_dict = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    model_performance[name] = {'model': model, 'rmse': rmse, 'r2': r2}
    y_pred_dict[name] = y_pred
```

#### 7. Selecting the Best Model

**Purpose:** Identify the model with the lowest RMSE, which is considered the best performing model.

```python
best_model_name = min(model_performance, key=lambda k: model_performance[k]['rmse'])
best_model = model_performance[best_model_name]['model']
```

#### 8. Making Predictions

**Purpose:** Use the best-performing model to predict house prices in the test dataset and save these predictions in a CSV file (`submission.csv`).

```python
test_predictions = best_model.predict(test_data)
submission = pd.DataFrame({'Id': test_data.index + 1461, 'SalePrice': test_predictions})
submission.to_csv('/workspaces/Housing-Sales/submission.csv', index=False)
```

#### 9. Creating the Dashboard Layout

**Purpose:** Define the structure of the dashboard, including dropdowns for model selection, tables for displaying data, and graphs for visualizing results.

- **Dropdown:** Allows users to select a model to view its performance.
- **Scatter Plot:** Displays actual vs. predicted prices for the selected model.
- **Tables:** Display model performance metrics and a comparison of actual vs. predicted prices.

```python
app.layout = html.Div([
    html.H1("House Prices Dashboard", className="text-center mb-4 mt-4"),
    
    html.Div([
        html.Label('Select a Model:'),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': name, 'value': name} for name in models.keys()],
            value=best_model_name
        ),
        dcc.Graph(id='prediction-actual-plot'),
        html.Div(id='model-performance-metrics'),
        
        html.H5(id='prediction-table-heading', className="text-center mt-4 mb-2"),
        
        dash_table.DataTable(
            id='prediction-table',
            columns=[
                {'name': 'ID', 'id': 'Id'},
                {'name': 'Actual Price', 'id': 'ActualPrice'},
               

 {'name': 'Predicted Price', 'id': 'PredictedPrice'},
                {'name': 'Estimate/Actual (%)', 'id': 'Percentage'}
            ],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
        ),
    ]),

    html.Hr(style={'border': '3px solid black'}),
    
    html.Div([
        html.H5("Model Summary", className="text-center mb-4"),
        dash_table.DataTable(
            id='models-table',
            columns=[{"name": i, "id": i} for i in ['Model', 'RMSE', 'R^2', 'Best Model']],
            data=model_table_data,
            page_size=len(models),
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Best Model} = "Yes"', 'column_id': 'Best Model'},
                    'backgroundColor': 'dodgerblue', 'color': 'white',
                },
                {
                    'if': {'filter_query': f'{{RMSE}} = {best_rmse:.8f}', 'column_id': 'RMSE'},
                    'backgroundColor': 'lightgreen', 'color': 'black',
                }
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
        ),
        html.P("This table shows the RMSE and R² for each model. The best model is highlighted.")
    ]),

    html.Hr(style={'border': '2px solid gray'}),
    html.H3("Would you like to download the final results (submission.csv)?", className="text-center mt-4 mb-2"),
    html.Div(className="text-center", children=[
        html.Button("Download", id="download-button", n_clicks=0, className="text-center")
    ]),
    dcc.Download(id="download-link")
])
```

#### 10. Dynamic Updates with Callbacks

**Purpose:** Update the dashboard dynamically based on user interactions. When a user selects a model from the dropdown, the corresponding scatter plot, performance metrics, and prediction table are updated.

- **Callback Functions:** These functions listen for user input (like selecting a model) and update the relevant components of the dashboard accordingly.

```python
@app.callback(
    [Output('prediction-actual-plot', 'figure'),
     Output('model-performance-metrics', 'children'),
     Output('prediction-table', 'data'),
     Output('prediction-table-heading', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_model_visualization(selected_model):
    y_pred_selected = y_pred_dict[selected_model]
    rmse = model_performance[selected_model]['rmse']
    r2 = model_performance[selected_model]['r2']

    slope, intercept = np.polyfit(y_val, y_pred_selected, 1)
    trendline_eq = f"y = {slope:.2f}x + {intercept:.2f}"

    fig = px.scatter(x=y_val, y=y_pred_selected, labels={'x': 'Actual Sale Price', 'y': 'Predicted Sale Price'},
                     title=f"Actual vs Predicted Sale Prices ({selected_model})<br>Trendline: {trendline_eq}")
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=min(y_val), y0=min(y_val),
        x1=max(y_val), y1=max(y_val)
    )

    metrics = [
        html.P(f"Model: {selected_model}"),
        html.P(f"RMSE: {rmse:.8f}"),
        html.P(f"R^2: {r2:.3f}")
    ]
    
    table_data = []
    for idx in range(10):
        actual_price = round(y_val.iloc[idx])
        predicted_price = round(y_pred_selected[idx])
        percentage = round((predicted_price / actual_price) * 100) if actual_price != 0 else 0
        table_data.append({
            'Id': X_val.index[idx],
            'ActualPrice': actual_price,
            'PredictedPrice': predicted_price,
            'Percentage': f"{percentage}%"
        })
    
    heading = f"10 Example IDs of Actual vs Predicted for {selected_model}"

    return fig, metrics, table_data, heading
```

#### 11. Download Feature

**Purpose:** Allow users to download the final predictions (`submission.csv`). A button is provided that, when clicked, triggers the download.

- **Download Button:** Triggers the download when clicked.

```python
@app.callback(
    Output("download-link", "data"),
    [Input("download-button", "n_clicks")],
    prevent_initial_call=True
)
def download_submission(n_clicks):
    if n_clicks > 0:
        return dcc.send_file("/workspaces/Housing-Sales/submission.csv")
```

### Running the App

1. Ensure you have the required datasets (`train.csv` and `test.csv`) in the specified locations.
2. Run the script in a Python environment.
3. Open your web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard.

### Conclusion

This script creates a powerful, interactive dashboard for comparing machine learning models in predicting house prices. Users can easily select different models, visualize their performance, and download the final predictions for further use. The step-by-step breakdown provided in this README should help users understand the purpose and functionality of each part of the script.