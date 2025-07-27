from flask import Flask, request, jsonify 
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import traceback
import joblib

app = Flask(__name__)

# CORS for frontend (you should update the domain to your actual frontend URL in production)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],  # Replace with frontend domain if needed
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load models
try:
    xgb_model_path = os.path.join(os.path.dirname(__file__), "xgboost_irrigation_model.pkl")
    with open(xgb_model_path, "rb") as file:
        xgb_model = pickle.load(file)
    print("XGBoost model loaded successfully.")
except Exception as e:
    print("Error loading XGBoost model:", e)
    xgb_model = None

try:
    lstm_model_path = os.path.join(os.path.dirname(__file__), "improved_lstm_etc_model.h5")
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    print("LSTM model loaded successfully.")
except Exception as e:
    print("Error loading LSTM model:", e)
    lstm_model = None

try:
    q_table_path = os.path.join(os.path.dirname(__file__), "q_table.npy")
    Q_table = np.load(q_table_path)
    print("Q-learning model (Q-table) loaded successfully.")
except Exception as e:
    print("Error loading Q-table:", e)
    Q_table = None

try:
    scaler_X = joblib.load(os.path.join(os.path.dirname(__file__), "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(os.path.dirname(__file__), "scaler_y.pkl"))
    print("Scalers loaded successfully.")
except Exception as e:
    print("Error loading scalers:", e)
    scaler_X, scaler_y = None, None

FEATURE_COLUMNS = ["Min Temp", "Max Temp", "Humidity", "Kc", "Temp_Diff", "Humidity_Squared"]

temp_bins = np.linspace(-5, 50, num=5)
humidity_bins = np.linspace(0, 100, num=5)
kc_bins = np.linspace(0, 1.5, num=5)

@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return "", 204
    return "Irrigation Prediction API is Running!"

@app.route("/predict-xgb", methods=["POST", "OPTIONS"])
def predict_xgb():
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        if xgb_model is None:
            return jsonify({"error": "XGBoost model failed to load."}), 500

        data = request.json
        missing_features = [col for col in FEATURE_COLUMNS[:4] if col not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        input_data = [data[col] for col in FEATURE_COLUMNS[:4]]
        input_data = np.array([input_data], dtype=np.float32)

        etc_prediction = xgb_model.predict(input_data)[0]

        return jsonify({"prediction": float(etc_prediction)})

    except Exception as e:
        print("Error in /predict-xgb:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/predict-lstm", methods=["POST", "OPTIONS"])
def predict_lstm():
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        if lstm_model is None:
            return jsonify({"error": "LSTM model failed to load."}), 500
        if scaler_X is None or scaler_y is None:
            return jsonify({"error": "Scalers failed to load."}), 500

        data = request.json
        required_inputs = ["Min Temp", "Max Temp", "Humidity", "Kc"]
        missing_features = [col for col in required_inputs if col not in data]

        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        temp_diff = data["Max Temp"] - data["Min Temp"]
        humidity_squared = data["Humidity"] ** 2

        input_data = np.array([[data["Min Temp"], data["Max Temp"], data["Humidity"], data["Kc"], temp_diff, humidity_squared]])

        input_scaled = scaler_X.transform(input_data)

        input_padded = np.zeros((1, 10, len(FEATURE_COLUMNS)))
        input_padded[:, -1, :] = input_scaled

        prediction_scaled = lstm_model.predict(input_padded)[0][0]
        predicted_ETc = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

        return jsonify({"prediction": float(predicted_ETc)})

    except Exception as e:
        print("Error in /predict-lstm:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def predict_etc_qlearning(min_temp, max_temp, humidity, kc):
    try:
        temp_idx = np.digitize((min_temp + max_temp) / 2, temp_bins) - 1
        humidity_idx = np.digitize(humidity, humidity_bins) - 1
        kc_idx = np.digitize(kc, kc_bins) - 1

        temp_idx = np.clip(temp_idx, 0, len(temp_bins) - 2)
        humidity_idx = np.clip(humidity_idx, 0, len(humidity_bins) - 2)
        kc_idx = np.clip(kc_idx, 0, len(kc_bins) - 2)

        state = (temp_idx, humidity_idx, kc_idx)
        etc_prediction = np.median(Q_table[state])
        etc_prediction = np.clip(etc_prediction, 0, np.max(Q_table))

        if scaler_y is not None:
            etc_prediction = scaler_y.inverse_transform([[etc_prediction]])[0][0]

        return etc_prediction

    except Exception as e:
        print("Error in predict_etc_qlearning:", e)
        return None

@app.route("/predict-qlearning", methods=["POST", "OPTIONS"])
def predict_qlearning():
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        if Q_table is None:
            return jsonify({"error": "Q-learning model failed to load."}), 500

        data = request.json
        min_temp = data.get("Min Temp")
        max_temp = data.get("Max Temp")
        humidity = data.get("Humidity")
        kc = data.get("Kc")

        if None in [min_temp, max_temp, humidity, kc]:
            return jsonify({"error": "Missing input values"}), 400

        etc_prediction = predict_etc_qlearning(min_temp, max_temp, humidity, kc)

        return jsonify({"prediction": float(etc_prediction)})

    except Exception as e:
        print("Error in /predict-qlearning:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# For mod_wsgi compatibility
application = app
