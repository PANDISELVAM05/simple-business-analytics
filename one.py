import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from prophet import Prophet

# Step 1: Load Data
data = pd.read_csv('sales_data.csv')

data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

target_variable = 'Monthly Sales (USD)'
features = ['Marketing Spend (USD)', 'Promotions', 'Season']

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Step 2: Forecasting Methods
# 2.1 Moving Average
def moving_average_forecast(data, window):
    return data[target_variable].rolling(window=window).mean()

ma_3 = moving_average_forecast(data, window=3)
ma_6 = moving_average_forecast(data, window=6)

# 2.2 Exponential Smoothing
hw_model = ExponentialSmoothing(train[target_variable], trend='add', seasonal='add', seasonal_periods=12)
hw_fit = hw_model.fit()
hw_forecast = hw_fit.forecast(steps=len(test))

# 2.3 ARIMA
arima_model = ARIMA(train[target_variable], order=(1, 1, 1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))

# 2.4 Prophet
prophet_data = train.reset_index().rename(columns={"Month": "ds", target_variable: "y"})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
prophet_forecast = prophet_model.predict(future)[['ds', 'yhat']]

# 2.5 Linear Regression for Time Series
X_train_time = np.arange(len(train)).reshape(-1, 1)
y_train_time = train[target_variable]
X_test_time = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

linear_model = LinearRegression()
linear_model.fit(X_train_time, y_train_time)
linear_forecast = linear_model.predict(X_test_time)

# Step 3: Machine Learning Models
# Preprocessing for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Marketing Spend (USD)', 'Promotions']),
        ('cat', OneHotEncoder(), ['Season'])
    ]
)

# Define pipelines for each model
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge())])
lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso())])
svr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', SVR(kernel='rbf'))])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor())])
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor())])
nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))])

# Train-test split
X = data[features]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
ridge_pipeline.fit(X_train, y_train)
ridge_pred = ridge_pipeline.predict(X_test)

lasso_pipeline.fit(X_train, y_train)
lasso_pred = lasso_pipeline.predict(X_test)

svr_pipeline.fit(X_train, y_train)
svr_pred = svr_pipeline.predict(X_test)

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

gb_pipeline.fit(X_train, y_train)
gb_pred = gb_pipeline.predict(X_test)

nn_pipeline.fit(X_train, y_train)
nn_pred = nn_pipeline.predict(X_test)

# Step 4: Evaluation and Comparative Analysis
def evaluate_model(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# Evaluate all models
results = {
    'Ridge Regression': evaluate_model(y_test, ridge_pred),
    'Lasso Regression': evaluate_model(y_test, lasso_pred),
    'SVR': evaluate_model(y_test, svr_pred),
    'Random Forest': evaluate_model(y_test, rf_pred),
    'Gradient Boosting': evaluate_model(y_test, gb_pred),
    'Neural Network': evaluate_model(y_test, nn_pred),
}

# Print Comparative Results
for model, metrics in results.items():
    print(f"{model}: {metrics}")

# Step 5: Generate Report
# Generate a structured report including visualizations
report_file = "analysis_report.txt"

with open(report_file, "w") as file:
    file.write("Performance Comparison:\n")
    for model, metrics in results.items():
        file.write(f"{model}: {metrics}\n")

    file.write("\nKey Findings:\n")
    file.write("1. Overall trends and patterns observed in the data.\n")
    file.write("2. Important features influencing the target variable.\n")
    file.write("3. Forecasted values for the next 12 months (using the best model).\n")

# Generate visualizations
plt.figure(figsize=(12, 6))
plt.plot(test[target_variable].values, label="Actual", marker='o')
plt.plot(hw_forecast, label="Exponential Smoothing", marker='o')
plt.plot(arima_forecast, label="ARIMA", marker='o')
plt.plot(linear_forecast, label="Linear Regression", marker='o')
plt.title("Forecasting Results")
plt.legend()
plt.savefig("forecasting_results.png")

plt.figure(figsize=(12, 6))
models = list(results.keys())
rmses = [metrics['RMSE'] for metrics in results.values()]
plt.bar(models, rmses, color='skyblue')
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.xticks(rotation=45)
plt.savefig("model_comparison.png")

print(f"Report and visualizations saved: {report_file}, forecasting_results.png, model_comparison.png")
