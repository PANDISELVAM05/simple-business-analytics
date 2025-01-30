from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Route for rendering the HTML page
@app.route("/")
def home():
    return render_template("main.html")

# Route for uploading and processing the file
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    # Save the uploaded file
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Load and process the file
    try:
        data = pd.read_csv(filepath)
        # Example: Return the first 5 rows of the dataset
        result = data.head().to_json()
        return jsonify({"message": "File processed successfully", "data": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
