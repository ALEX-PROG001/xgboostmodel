import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

# --- Load Model & Scalers ---
base_path = os.path.dirname(__file__)  
model_path = os.path.join(base_path, "improved_lstm_etc_model.h5")
scaler_X_path = os.path.join(base_path, "scaler_X.pkl")
scaler_y_path = os.path.join(base_path, "scaler_y.pkl")

model = load_model(model_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# --- Define New Input Data ---
new_data = pd.DataFrame([{
    'Min Temp': 15.0,
    'Max Temp': 28.0,
    'Humidity': 70.0,
    'Kc': 0.85
}])

# --- Feature Engineering (Ensure Features Match Model) ---
new_data["Temp_Diff"] = new_data["Max Temp"] - new_data["Min Temp"]
new_data["Humidity_Squared"] = new_data["Humidity"] ** 2

# Reorder columns to match training features
feature_columns = ["Min Temp", "Max Temp", "Humidity", "Kc", "Temp_Diff", "Humidity_Squared"]
new_data = new_data[feature_columns]

# --- Scale Data ---
scaled_new_data = scaler_X.transform(new_data)

# --- Reshape Data for LSTM (10 time steps needed) ---
X_new = np.tile(scaled_new_data, (10, 1))  # Repeat row 10 times
X_new = np.expand_dims(X_new, axis=0)  # Reshape to (1, 10, features)

# --- Predict ETc ---
pred_scaled = model.predict(X_new)

# --- Reverse Scaling ---
pred_etc = scaler_y.inverse_transform(pred_scaled)[0][0]

print(f" Predicted ETc: {pred_etc:.2f} mm/day")
