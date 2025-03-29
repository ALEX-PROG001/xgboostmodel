import pandas as pd
import joblib

# Load the saved model
model = joblib.load("xgboost_irrigation_model.pkl")

# Define feature columns used in training
feature_columns = ["Min Temp", "Max Temp", "Humidity", "Kc"]

# Create a test dataset with values for prediction
test_data = pd.DataFrame({
    "Min Temp": [9.0, 9.2],
    "Max Temp": [28.0, 29.0],
    "Humidity": [75, 74],
    "Kc": [0.65, 0.70]
})

# Convert to numeric (if necessary)
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Make predictions
predictions = model.predict(test_data)

# Print predictions
print("Predicted ETc:", predictions)
