import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('lstm_irrigation.h5')

# Load scaler
scaler = joblib.load('scaler.pkl')

# Load original feature order
original_feature_order = scaler.feature_names_in_.tolist()

# Define new input data (single row)
new_data = pd.DataFrame([{
    'Kc': 0,
    'ETc_day': 5.5,
    'ETc_dec': 50,
    'EffRain': 8,
    'IrrReq': 0,
    'Min Temp': 0,
    'Max Temp': 0,
    'Humidity': 0,
    'Wind': 0,
    'Sun': 0,
    'Rad': 0,
    'ETo': 0
}])

# Reorder to match training feature order
new_data = new_data[original_feature_order]

# Scale data
scaled_new_data = scaler.transform(new_data)

# **Fix**: Repeat the data 10 times to create a valid sequence for LSTM
X_new = np.tile(scaled_new_data, (10, 1))  # Shape becomes (10, 12)

# Reshape to match LSTM expected input shape: (1, 10, 12)
X_new = np.expand_dims(X_new, axis=0)

# Predict irrigation requirement
pred_scaled = model.predict(X_new)

# Reverse scaling
pred_irrigation = scaler.inverse_transform(
    np.hstack([pred_scaled, np.zeros((pred_scaled.shape[0], scaled_new_data.shape[1] - 1))])
)[:, 0][0]

print(f"Predicted Irrigation Requirement: {pred_irrigation:.2f} mm/dec")
