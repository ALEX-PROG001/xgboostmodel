from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import traceback

app = Flask(__name__)

# Load XGBoost Model
try:
    xgb_model_path = os.path.join(os.path.dirname(__file__), "xgboost_irrigation_model.pkl")
    with open(xgb_model_path, "rb") as file:
        xgb_model = pickle.load(file)
    print("XGBoost model loaded successfully.")
except Exception as e:
    print("Error loading XGBoost model:", e)
    xgb_model = None

# Load LSTM Model
try:
    lstm_model_path = os.path.join(os.path.dirname(__file__), "lstm_irrigation.h5")
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    print("LSTM model loaded successfully.")
except Exception as e:
    print("Error loading LSTM model:", e)
    lstm_model = None

# Load Q-learning Q-table
try:
    q_table_path = os.path.join(os.path.dirname(__file__), "q_table.npy")
    Q_table = np.load(q_table_path)
    print("Q-learning model (Q-table) loaded successfully.")
except Exception as e:
    print("Error loading Q-table:", e)
    Q_table = None

# Feature Columns
FEATURE_COLUMNS = [
    "Kc", "ETc (mm/day)", "ETc (mm/dec)", "Eff rain (mm/dec)",
    "Min Temp", "Max Temp", "Humidity", "Wind", "Sun", "Rad", "ETo"
]

# Define Bins for Q-learning (Same as Training)
temp_bins = np.linspace(-5, 40, num=5)
humidity_bins = np.linspace(0, 100, num=5)
kc_bins = np.linspace(0, 1.2, num=5)

@app.route("/", methods=["GET"])
def home():
    return "Irrigation Prediction API is Running!"

@app.route("/predict-xgb", methods=["POST"])
def predict_xgb():
    try:
        if xgb_model is None:
            return jsonify({"error": "XGBoost model failed to load."}), 500

        data = request.json
        missing_features = [col for col in FEATURE_COLUMNS if col not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        input_data = [data[col] for col in FEATURE_COLUMNS]
        input_data = np.array([input_data], dtype=np.float32)

        irrigation_req = xgb_model.predict(input_data)[0]

        return jsonify({"model": "xgboost", "irrigation_requirement": float(irrigation_req)})

    except Exception as e:
        print("Error in /predict-xgb:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/predict-lstm", methods=["POST"])
def predict_lstm():
    try:
        if lstm_model is None:
            return jsonify({"error": "LSTM model failed to load."}), 500

        data = request.json
        input_data = [data[col] for col in FEATURE_COLUMNS]
        if len(input_data) != 12:
            return jsonify({"error": f"Expected 12 features, but got {len(input_data)}"}), 400

        input_data = np.array([input_data], dtype=np.float32)
        input_data = np.tile(input_data, (1, 10, 1))

        irrigation_req = lstm_model.predict(input_data)[0][0]

        return jsonify({"model": "lstm", "irrigation_requirement": float(irrigation_req)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def recommend_irrigation(min_temp, humidity, kc):
    state = (
        np.digitize(min_temp, temp_bins, right=True),
        np.digitize(humidity, humidity_bins, right=True),
        np.digitize(kc, kc_bins, right=True),
    )
    action = np.argmax(Q_table[state])
    return action

@app.route("/predict-qlearning", methods=["POST"])
def predict_qlearning():
    try:
        if Q_table is None:
            return jsonify({"error": "Q-learning model failed to load."}), 500

        data = request.json
        min_temp = data.get("min_temp")
        humidity = data.get("humidity")
        kc = data.get("kc")

        if min_temp is None or humidity is None or kc is None:
            return jsonify({"error": "Missing input values"}), 400

        irrigation_prediction = recommend_irrigation(min_temp, humidity, kc)

        return jsonify({"model": "q-learning", "irrigation_requirement": int(irrigation_prediction)})

    except Exception as e:
        print("Error in /predict-qlearning:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
