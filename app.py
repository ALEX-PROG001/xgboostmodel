from flask import Flask, request, jsonify
import pickle
import pandas as pd  # Required for XGBoost
import numpy as np
import os


# Load the trained XGBoost model
model_path = os.path.join(os.path.dirname(__file__), "xgboost_irrigation_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Extract required features (all 11 features)
        feature_columns = [
            "Kc", "ETc (mm/day)", "ETc (mm/dec)", "Eff rain (mm/dec)",
            "Min Temp", "Max Temp", "Humidity", "Wind", "Sun", "Rad", "ETo"
        ]

        input_data = [data[col] for col in feature_columns]

        # Convert to NumPy array and reshape for model
        input_data = np.array([input_data], dtype=np.float32)

        # Make prediction
        irrigation_req = model.predict(input_data)[0]

        # Convert float32 to a standard float
        irrigation_req = float(irrigation_req)

        return jsonify({"irrigation_requirement": irrigation_req})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route("/", methods=["GET"])
def home():
    return "Irrigation Prediction API is Running!"


if __name__ == "__main__":
    app.run(debug=True)
