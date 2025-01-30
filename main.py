from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from prophet import Prophet
import os
import shutil
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'csv'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the path to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def index():
    return render_template('main.html')

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Call the existing analysis function
        results, visualizations = perform_analysis(filepath)
        
        # Return the analysis results as a JSON response
        return jsonify(results=results, visualizations=visualizations)

    return jsonify({"error": "Invalid file type. Please upload a CSV file."})

# Perform analysis function (integrating your existing code here)
def perform_analysis(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)

    # Core analytics code (same as your existing code)
    target_variable = 'Monthly Sales (USD)'
    features = ['Marketing Spend (USD)', 'Promotions', 'Season']
    train, test = train_test_split(data, test_size=0.2, shuffle=False)

    # Forecasting Methods
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

    # Machine Learning Models
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Marketing Spend (USD)', 'Promotions']),
            ('cat', OneHotEncoder(), ['Season'])
        ]
    )

    models = {
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "SVR": SVR(kernel='rbf'),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    results = {}
    X = data[features]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

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
    models_list = list(results.keys())
    rmses = [metrics['RMSE'] for metrics in results.values()]
    plt.bar(models_list, rmses, color='skyblue')
    plt.title("Model RMSE Comparison")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.savefig("model_comparison.png")

    # Define the local download folder (change `YourUsername` if needed)
    DOWNLOAD_FOLDER = "C:/Users/Pandi selvam/Downloads/"

    # Ensure the folder exists
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    # Save results as a text file
    report_path = os.path.join(DOWNLOAD_FOLDER, "analysis_report.txt")
    with open(report_path, "w") as file:
        file.write("Performance Comparison:\n")
        for model, metrics in results.items():
            file.write(f"{model}:\n")
            file.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
            file.write(f"  MAE: {metrics['MAE']:.2f}\n")
            file.write(f"  R2: {metrics['R2']:.2f}\n")
            file.write("\n")

    # Move the images to the Downloads folder
    shutil.move("forecasting_results.png", os.path.join(DOWNLOAD_FOLDER, "forecasting_results.png"))
    shutil.move("model_comparison.png", os.path.join(DOWNLOAD_FOLDER, "model_comparison.png"))

    return results, {
        "forecasting_results": "forecasting_results.png",
        "model_comparison": "model_comparison.png",
        "report": report_path
    }

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
