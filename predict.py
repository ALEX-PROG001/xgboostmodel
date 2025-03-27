import pandas as pd
import joblib

# Load the saved model
model = joblib.load("xgboost_irrigation_model.pkl")

# Define your feature columns (should match what was used in training)
feature_columns = [
    "Kc",
    "ETc (mm/day)",
    "ETc (mm/dec)",
    "Eff rain (mm/dec)",
    "Min Temp",
    "Max Temp",
    "Humidity",
    "Wind",
    "Sun",
    "Rad",
    "ETo"
]

#  Create a new DataFrame with future data for prediction.
# Replace the values below with your actual forecast data.
new_data = pd.DataFrame({
    "Kc": [0.65, 0.70],
    "ETc (mm/day)": [2.5, 2.7],
    "ETc (mm/dec)": [15.0, 16.0],
    "Eff rain (mm/dec)": [12.0, 13.0],
    "Min Temp": [9.0, 9.2],
    "Max Temp": [28.0, 29.0],
    "Humidity": [75, 74],
    "Wind": [23, 24],
    "Sun": [3, 3.2],
    "Rad": [7.5, 7.8],
    "ETo": [1.3, 1.4]
})

# (Optional) Convert feature columns to numeric, if necessary
for col in feature_columns:
    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

# Predict irrigation requirements (mm/dec)
predictions = model.predict(new_data[feature_columns])
print("Predicted Irrigation Requirements (mm/dec):", predictions)
