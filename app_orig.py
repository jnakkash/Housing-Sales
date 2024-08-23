import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error

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
test_ids = test_df['Id']
X_test = test_df.drop(['Id'], axis=1)
X_test_preprocessed = preprocessor.transform(X_test)
test_preds = rf.predict(X_test_preprocessed)
test_preds_exp = np.expm1(test_preds)  # Reverse the log1p transformation

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("House Prices Prediction Dashboard"),

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
        dcc.Graph(id='importance-plot'),
        html.P(
            "This plot ranks the features (characteristics of houses) that the model thinks are the most important "
            "in predicting house prices. You can use the slider to select how many of the top features you want to see."
        )
    ], style={'padding': '20px'}),

    html.Div([
        html.Label("Select Distribution to View:"),
        dcc.Dropdown(
            id='distribution-dropdown',
            options=[
                {'label': 'Validation Set', 'value': 'val'},
                {'label': 'Test Set', 'value': 'test'}
            ],
            value='val'
        ),
        dcc.Graph(id='distribution-plot'),
        html.P(
            "This plot compares the distribution (spread) of actual house prices to the prices predicted by the model. "
            "It shows how well the model’s predictions match the real prices. Use the dropdown to switch between the validation set "
            "and the test set."
        )
    ], style={'padding': '20px'}),

    html.Div([
        dcc.Graph(id='residuals-plot'),
        html.P(
            "This scatter plot shows the differences between the actual prices and the predicted prices, called residuals. "
            "Each dot represents one house. If the dot is near the horizontal line, the prediction was close to the actual price."
        )
    ], style={'padding': '20px'}),

    html.Div([
        dcc.Graph(id='heatmap-plot'),
        html.P(
            "This heatmap shows how different features (like house size, number of rooms, etc.) relate to each other. "
            "If two features are highly correlated, they will have a strong relationship."
        )
    ], style={'padding': '20px'}),

    html.Div([
        html.H3("Model Performance Metrics"),
        html.P(id="metrics-output"),
        html.P(
            "This section shows some key numbers that summarize how well the model is doing. "
            "RMSLE tells us how far off the model’s predictions are in a way that treats small and large errors more equally. "
            "RMSE is another measure of error, and R² indicates how much of the price variation the model can explain."
        )
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9'})
])

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
        fig.update_layout(title='Distribution of Final Predicted Prices on Test Set')
    return fig

@app.callback(
    Output('residuals-plot', 'figure'),
    Input('distribution-dropdown', 'value'))
def update_residuals_plot(dataset):
    if dataset == 'val':
        fig = px.scatter(x=val_preds, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                         title='Residuals Plot')
    elif dataset == 'test':
        fig = px.scatter(x=test_preds_exp, y=np.zeros_like(test_preds_exp), labels={'x': 'Predicted Values', 'y': 'Residuals'},
                         title='Test Set Predictions (No Residuals Available)')
    return fig

@app.callback(
    Output('heatmap-plot', 'figure'),
    Input('feature-slider', 'value'))
def update_heatmap_plot(top_n):
    top_features = importance_df.head(top_n)['Feature'].values
    correlation_matrix = pd.DataFrame(X_train, columns=feature_names).loc[:, top_features].corr()
    fig = px.imshow(correlation_matrix, title='Feature Correlation Heatmap')
    return fig

@app.callback(
    Output('metrics-output', 'children'),
    Input('distribution-dropdown', 'value'))
def update_metrics_output(dataset):
    if dataset == 'val':
        return [
            f"RMSLE: {rmsle:.4f}",
            html.Br(),
            f"RMSE: {rmse:.4f}",
            html.Br(),
            f"R²: {r2:.4f}"
        ]
    elif dataset == 'test':
        return "Model performance metrics are only available for the validation set."

if __name__ == '__main__':
    app.run_server(debug=True)



