import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import numpy as np

# Correct file paths
train_file_path = '/workspaces/Housing-Sales/train.csv'
test_file_path = '/workspaces/Housing-Sales/test.csv'

# Load data
train_df = pd.read_csv(train_file_path)

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
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # Ensure dense output

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply transformations
X_preprocessed = preprocessor.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Get the feature names after transformation
encoded_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, encoded_features])

# Train a RandomForest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Predictions and metrics on validation set
val_preds = rf.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(y_val, val_preds))
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2 = r2_score(y_val, val_preds)

# Residuals for visualization
residuals = y_val - val_preds

# Predictions on the test set
test_df = pd.read_csv(test_file_path)
X_test = test_df.drop(['Id'], axis=1)
X_test_preprocessed = preprocessor.transform(X_test)
test_preds = rf.predict(X_test_preprocessed)
test_preds_exp = np.expm1(test_preds)  # Reverse the log1p transformation

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("House Prices Prediction Dashboard"),

    # Dropdown for selecting a feature to plot against SalePrice
    html.Div([
        html.Label("Select Feature to Plot Against SalePrice:"),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in numeric_features],
            value='GrLivArea'  # Default value
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Scatter plot based on the selected feature
    dcc.Graph(id='scatter-plot'),
    html.P(
        "This scatter plot shows the relationship between a selected feature of a house (like its size or the number "
        "of rooms) and its sale price. You can choose a different feature from the dropdown menu to see how it impacts "
        "the house price."
    ),

    # Slider for selecting the number of top features to display
    html.Div([
        html.Label("Select Number of Top Features:"),
        dcc.Slider(
            id='feature-slider',
            min=5,
            max=20,
            step=1,
            value=10,
            marks={i: f'{i}' for i in range(5, 21)},
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    # Bar plot for feature importance
    dcc.Graph(id='importance-plot'),
    html.P(
        "This bar plot shows the top features that our model found to be the most important in predicting house prices. "
        "You can use the slider to adjust the number of features shown."
    ),

    # Histogram for SalePrice distribution with dropdown for dataset selection
    html.Div([
        html.Label("Select Dataset for Distribution Plot:"),
        dcc.Dropdown(
            id='distribution-dropdown',
            options=[
                {'label': 'Validation Set', 'value': 'val'},
                {'label': 'Test Set', 'value': 'test'}
            ],
            value='val'  # Default value
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='distribution-plot'),
    html.P(
        "This histogram compares the distribution of actual house prices and the prices predicted by the model. "
        "You can switch between the validation set (used for testing the model during development) and the test set "
        "(used for final evaluation)."
    ),

    # Residual plot
    dcc.Graph(id='residuals-plot'),
    html.P(
        "This scatter plot shows the residuals, which are the differences between the actual prices and the predicted prices. "
        "If the dots are close to the horizontal line, it means the model made accurate predictions. Large deviations indicate "
        "where the model's predictions were off."
    ),

    # Correlation Heatmap
    dcc.Graph(id='heatmap'),
    html.P(
        "This heatmap shows how different features are related to each other. For example, it can show if houses with more rooms "
        "tend to also have larger living areas. High correlation between features can provide insights into what drives house prices."
    ),

    # Model performance metrics
    html.Div([
        html.H3("Model Performance Metrics"),
        html.P(id="metrics-output"),
        html.P(
            "This section displays key metrics that summarize how well the model is performing: "
            "- RMSLE (Root Mean Squared Logarithmic Error) measures the accuracy of predictions, "
            "giving more importance to errors on cheaper houses. "
            "- RMSE (Root Mean Squared Error) measures the average magnitude of prediction errors. "
            "- R² (R-squared) indicates how much of the variance in house prices the model can explain."
        )
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9'}),

    # Final Summary of the Model and Results
    html.Div([
        html.H2("Final Model Summary and Results"),
        html.P(
            f"The best model in this analysis is a Random Forest Regressor, which was trained using "
            f"a dataset of house features. The model achieved an RMSLE of {rmsle:.4f}, indicating that it can predict "
            "house prices with a reasonable degree of accuracy. The final predictions on the test set have been generated "
            "and are available for submission."
        ),
        html.P(
            "The model's predictions are generally accurate, but there are some cases where the predictions deviate from "
            "the actual prices. These deviations are shown in the residuals plot, and they highlight areas where the model "
            "could be further improved, possibly by adding more features or using a more complex model."
        )
    ])
])

# Callbacks for updating the plots based on user interaction

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('feature-dropdown', 'value'))
def update_scatter_plot(selected_feature):
    fig = px.scatter(train_df, x=selected_feature, y='SalePrice', title=f'SalePrice vs {selected_feature}',
                     labels={selected_feature: selected_feature, 'SalePrice': 'Sale Price'})
    return fig

@app.callback(
    Output('importance-plot', 'figure'),
    Input('feature-slider', 'value'))
def update_importance_plot(top_n):
    top_features_df = importance_df.head(top_n)
    fig = px.bar(top_features_df, x='Importance', y='Feature', orientation='h', title=f'Top {top_n} Feature Importances')
    return fig

@app.callback(
    Output('distribution-plot', 'figure'),
    Input('distribution-dropdown', 'value'))
def update_distribution_plot(dataset):
    if dataset == 'val':
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.expm1(y_val), name='Actual Prices', opacity=0.75))
        fig.add_trace(go.Histogram(x=np.expm1(val_preds), name='Predicted Prices', opacity=0.75))
        fig.update_layout(barmode='overlay', title='Distribution of Actual vs Predicted Prices (Validation Set)')
        fig.update_traces(opacity=0.6)
    elif dataset == 'test':
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=test_preds_exp, name='Predicted Prices', opacity=0.75))
        fig.update_layout(title='Distribution of Predicted Prices on Test Set')
    return fig