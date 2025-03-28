from flask import Flask, request, jsonify
import pickle
import pandas as pd  # Required for XGBoost
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Load XGBoost Model
xgb_model_path = os.path.join(os.path.dirname(__file__), "xgboost_irrigation_model.pkl")
with open(xgb_model_path, "rb") as file:
    xgb_model = pickle.load(file)

# Load LSTM Model
lstm_model_path = os.path.join(os.path.dirname(__file__), "lstm_irrigation_model.h5")
lstm_model = tf.keras.models.load_model(lstm_model_path)

# Feature columns expected in request JSON
FEATURE_COLUMNS = [
    "Kc", "ETc (mm/day)", "ETc (mm/dec)", "Eff rain (mm/dec)",
    "Min Temp", "Max Temp", "Humidity", "Wind", "Sun", "Rad", "ETo"
]

@app.route("/", methods=["GET"])
def home():
    return "Irrigation Prediction API is Running!"

@app.route("/predict-xgb", methods=["POST"])
def predict_xgb():
    try:
        data = request.json

        # Extract features
        input_data = [data[col] for col in FEATURE_COLUMNS]
        input_data = np.array([input_data], dtype=np.float32)

        # Predict using XGBoost
        irrigation_req = xgb_model.predict(input_data)[0]

        return jsonify({"model": "xgboost", "irrigation_requirement": float(irrigation_req)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-lstm", methods=["POST"])
def predict_lstm():
    try:
        data = request.json

        # Extract features
        input_data = [data[col] for col in FEATURE_COLUMNS]
        input_data = np.array([input_data], dtype=np.float32)

        # Reshape for LSTM (expects 3D shape: [batch_size, time_steps, features])
        input_data = input_data.reshape(1, 1, len(FEATURE_COLUMNS))

        # Predict using LSTM
        irrigation_req = lstm_model.predict(input_data)[0][0]

        return jsonify({"model": "lstm", "irrigation_requirement": float(irrigation_req)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
