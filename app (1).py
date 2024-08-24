from dash import Dash, dcc, html, dash_table  # Import necessary libraries for creating the dashboard
from dash.dependencies import Input, Output   # Import Input and Output components for dynamic updates
import pandas as pd                           # Import Pandas for data handling and manipulation
import plotly.express as px                   # Import Plotly Express for creating visualizations
import numpy as np                            # Import NumPy for numerical operations
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor)  # Import various regression models from sklearn
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)    # Import linear models from sklearn
from sklearn.tree import DecisionTreeRegressor # Import Decision Tree Regressor
from sklearn.svm import SVR                   # Import Support Vector Regressor
from sklearn.neighbors import KNeighborsRegressor  # Import K-Nearest Neighbors Regressor
from sklearn.model_selection import train_test_split # Import train_test_split for splitting data into training and validation sets
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics to evaluate model performance
from sklearn.impute import SimpleImputer      # Import SimpleImputer to handle missing data
from sklearn.pipeline import Pipeline         # Import Pipeline to streamline preprocessing and modeling steps
from sklearn.preprocessing import StandardScaler # Import StandardScaler for scaling features

# 1. Load the data
# This block attempts to load the training and testing datasets from CSV files.
# If the files can't be found, it throws an error and stops the program.
try:
    train_data = pd.read_csv('/workspaces/Housing-Sales/train.csv')  # Load the training data
    test_data = pd.read_csv('/workspaces/Housing-Sales/test.csv')    # Load the test data
except FileNotFoundError:
    print("Error: File not found. Please check the file paths.")  # Print an error if files are not found
    exit()

# 2. Check if the 'SalePrice' column is in the training data.
# 'SalePrice' is what we're trying to predict, so it must be present.
if 'SalePrice' not in train_data.columns:
    print("Error: 'SalePrice' column not found in training data.")  # Print an error if the SalePrice column is missing
    exit()

# 3. Data Preprocessing
# Preprocessing prepares the data for analysis and modeling. This involves:
# - Filling in missing values.
# - Dropping unnecessary columns.
# - Converting categorical (text) data into numbers.

# Fill missing values in the 'LotFrontage' column with the median value.
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].median())

# Drop columns that either have too much missing data or aren't useful for our model.
train_data = train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, errors='ignore')
test_data = test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, errors='ignore')

# Convert categorical variables (like text) into dummy variables (0s and 1s).
# This is necessary because models generally require numeric input.
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Ensure that the test dataset has the same columns as the training dataset.
# Any missing columns in the test set are filled with zeros.
X = train_data.drop('SalePrice', axis=1)  # Features (inputs) for training
y = train_data['SalePrice']  # Target (output) for training
test_data = test_data.reindex(columns=X.columns, fill_value=0)  # Align test data with training data

# 4. Split the data into training and validation sets.
# The training set is used to train the model, while the validation set is used to test it.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Define various models to train.
# Here, we set up different machine learning models that we'll train and compare.
# Each model is wrapped in a 'pipeline' which automates common preprocessing steps like filling missing values.

models = {
    'RandomForest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Random forest model
    ]),
    'GradientBoosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),  # Special model that handles missing data natively
    'LinearRegression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
        ('model', Ridge())
    ]),
    'Lasso': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.01, max_iter=10000))  # Lasso regression with specific parameters
    ]),
    'ElasticNet': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.01, max_iter=10000))
    ]),
    'DecisionTree': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeRegressor(random_state=42))  # Decision tree model
    ]),
    'SVR': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVR())  # Support Vector Regressor
    ]),
    'KNeighbors': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor())  # K-Nearest Neighbors
    ]),
    'AdaBoost': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', AdaBoostRegressor(random_state=42))  # AdaBoost model
    ])
}

# 6. Train each model and calculate performance metrics.
# This loop trains each model and then tests it on the validation set.
# The key performance metrics are RMSE (Root Mean Squared Error) and R^2 (R-squared).
model_performance = {}  # Dictionary to store model performance metrics
y_pred_dict = {}  # Store predictions for each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model with the training data
    y_pred = model.predict(X_val)  # Predict using the validation data
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))  # Calculate RMSE
    r2 = r2_score(y_val, y_pred)  # Calculate R-squared
    model_performance[name] = {'model': model, 'rmse': rmse, 'r2': r2}  # Store performance metrics
    y_pred_dict[name] = y_pred  # Store predictions for later use

# 7. Select the best model based on RMSE.
# The model with the lowest RMSE is considered the best.
best_model_name = min(model_performance, key=lambda k: model_performance[k]['rmse'])
best_model = model_performance[best_model_name]['model']
best_rmse = model_performance[best_model_name]['rmse']

# 8. Predict on the test set using the best model.
# Apply the best model to the test data to make final predictions.
test_predictions = best_model.predict(test_data)

# 9. Create a submission file for the test set predictions.
# This file can be submitted to a competition platform like Kaggle.
submission = pd.DataFrame({'Id': test_data.index + 1461, 'SalePrice': test_predictions})
submission.to_csv('/workspaces/Housing-Sales/submission.csv', index=False)

# 10. Prepare data for the model performance summary table.
# This table summarizes the performance (RMSE and R²) of each model.
model_table_data = [
    {'Model': name, 'RMSE': f"{performance['rmse']:.8f}", 'R^2': f"{performance['r2']:.3f}", 'Best Model': 'Yes' if name == best_model_name else 'No'}
    for name, performance in model_performance.items()
]

# 11. Initialize the Dash app.
# Dash is a framework that allows us to build web applications for data analysis.
app = Dash(__name__)
app.config.suppress_callback_exceptions = True  # Allow for callbacks to components that are created dynamically

# 12. Layout of the dashboard.
# This section defines the structure of the web application, including dropdowns, tables, and graphs.
app.layout = html.Div([
    html.H1("House Prices Dashboard", className="text-center mb-4 mt-4"),  # Title of the dashboard
    
    html.Div([
        html.Label('Select a Model:'),  # Dropdown menu label
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': name, 'value': name} for name in models.keys()],  # Dropdown options are the model names
            value=best_model_name  # Default selection is the best model
        ),
        dcc.Graph(id='prediction-actual-plot'),  # This will show a scatter plot of actual vs predicted prices
        html.Div(id='model-performance-metrics'),  # This will show RMSE and R² for the selected model
        
        # Heading for the table that compares actual and predicted prices
        html.H5(id='prediction-table-heading', className="text-center mt-4 mb-2"),
        
        # Table that shows the first 10 actual vs predicted prices, along with a percentage error
        dash_table.DataTable(
            id='prediction-table',
            columns=[
                {'name': 'ID', 'id': 'Id'},
                {'name': 'Actual Price', 'id': 'ActualPrice'},
                {'name': 'Predicted Price', 'id': 'PredictedPrice'},
                {'name': 'Estimate/Actual (%)', 'id': 'Percentage'}
            ],
            page_size=10,  # Display 10 rows at a time
            style_table={'overflowX': 'auto'},  # Allow horizontal scrolling if needed
            style_cell={
                'textAlign': 'center',  # Center-align text in the table
                'height': 'auto',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'whiteSpace': 'normal'
            },
        )
    ]),

    # Thicker and bolder line separator between sections
    html.Hr(style={'border': '3px solid black'}),

    # Model Summary section, which shows a summary of the performance of all models
    html.Div([
        html.H5("Model Summary", className="text-center mb-4"),  # Title of the summary section
        dash_table.DataTable(
            id='models-table',
            columns=[{"name": i, "id": i} for i in ['Model', 'RMSE', 'R^2', 'Best Model']],  # Columns for the summary table
            data=model_table_data,  # Data for the table
            page_size=len(models),  # Display all models in a single page
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Best Model} = "Yes"',  # Highlight the best model
                        'column_id': 'Best Model'
                    },
                    'backgroundColor': 'dodgerblue',
                    'color': 'white',
                },
                {
                    'if': {
                        'filter_query': f'{{RMSE}} = {best_rmse:.8f}',  # Highlight the row with the lowest RMSE
                        'column_id': 'RMSE'
                    },
                    'backgroundColor': 'lightgreen',
                    'color': 'black',
                }
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',  # Center-align text in the table
                'height': 'auto',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'whiteSpace': 'normal'
            },
        ),
        html.P("This table shows the RMSE and R² for each model. The best model is highlighted.")
    ]),

    html.Hr(style={'border': '2px solid gray'}),  # Another line separator

    # Download prompt and button
    html.H3("Would you like to download the final results (submission.csv)?", className="text-center mt-4 mb-2"),
    html.Div(className="text-center", children=[
        html.Button("Download", id="download-button", n_clicks=0, className="text-center")  # Button for downloading the CSV file
    ]),
    dcc.Download(id="download-link")  # Component to handle file download
])

# 13. Callbacks to update the dashboard based on user interactions.
# When a user selects a model from the dropdown, this function updates the scatter plot,
# performance metrics, and the table showing actual vs predicted prices.
@app.callback(
    [Output('prediction-actual-plot', 'figure'),
     Output('model-performance-metrics', 'children'),
     Output('prediction-table', 'data'),
     Output('prediction-table-heading', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_model_visualization(selected_model):
    # Get the predictions and performance metrics for the selected model.
    y_pred_selected = y_pred_dict[selected_model]
    rmse = model_performance[selected_model]['rmse']
    r2 = model_performance[selected_model]['r2']

    # Calculate a trendline (simple linear regression) to show the relationship between actual and predicted prices.
    slope, intercept = np.polyfit(y_val, y_pred_selected, 1)
    trendline_eq = f"y = {slope:.2f}x + {intercept:.2f}"

    # Generate a scatter plot of actual vs predicted prices with the trendline equation in the title.
    fig = px.scatter(x=y_val, y=y_pred_selected, labels={'x': 'Actual Sale Price', 'y': 'Predicted Sale Price'},
                     title=f"Actual vs Predicted Sale Prices ({selected_model})<br>Trendline: {trendline_eq}")
    fig.add_shape(
        type="line", line=dict(dash="dash"),  # Dashed line for the trendline
        x0=min(y_val), y0=min(y_val),
        x1=max(y_val), y1=max(y_val)
    )

    # Prepare the model performance metrics (RMSE and R²) to be displayed.
    metrics = [
        html.P(f"Model: {selected_model}"),
        html.P(f"RMSE: {rmse:.8f}"),  # Display RMSE with 8 decimal places
        html.P(f"R^2: {r2:.3f}")  # Display R² with 3 decimal places
    ]
    
    # Prepare data for the table that compares actual vs predicted prices.
    table_data = []
    for idx in range(10):
        actual_price = round(y_val.iloc[idx])  # Round the actual price to a whole number
        predicted_price = round(y_pred_selected[idx])  # Round the predicted price to a whole number
        percentage = round((predicted_price / actual_price) * 100) if actual_price != 0 else 0  # Calculate percentage error
        table_data.append({
            'Id': X_val.index[idx],  # Record ID
            'ActualPrice': actual_price,  # Actual price
            'PredictedPrice': predicted_price,  # Predicted price
            'Percentage': f"{percentage}%"  # Error as a percentage
        })
    
    # Set the heading for the prediction table.
    heading = f"10 Example IDs of Actual vs Predicted for {selected_model}"

    # Return the updated plot, metrics, table data, and heading.
    return fig, metrics, table_data, heading

# 14. Callback to handle the download request
# When the user clicks the "Download" button, this function triggers the download of the CSV file.
@app.callback(
    Output("download-link", "data"),
    [Input("download-button", "n_clicks")],
    prevent_initial_call=True
)
def download_submission(n_clicks):
    if n_clicks > 0:
        return dcc.send_file("/workspaces/Housing-Sales/submission.csv")

# 15. Run the app.
# This starts the web server, making the dashboard available in a web browser.
if __name__ == '__main__':
    app.run_server(debug=True)
