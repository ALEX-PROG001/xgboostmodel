import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Replace with your actual CSV file path (using raw string for Windows path)
csv_file_path = r"C:\Users\USER\Documents\irrigation model\irrigation model\climate_irrigation_data.csv"

# Columns from your screenshot:
# Month, Decade, Stage, Kc, ETc (mm/day), ETc (mm/dec), Eff rain (mm/dec), Irr. Req. (mm/dec),
# Min Temp, Max Temp, Humidity, Wind, Sun, Rad, ETo

# We will treat "Irr. Req. (mm/dec)" as our target variable.
# The features can include the rest of the numeric columns.
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

target_column = "Irr. Req. (mm/dec)"

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
try:
    data = pd.read_csv(csv_file_path, encoding='latin1')
    print("Data loaded successfully.")
except Exception as e:
    print("Error loading CSV file:", e)
    exit()

# Optional: Inspect the first few rows
print(data.head())

# ------------------------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------------------------
# 1. Drop rows with missing values in our feature or target columns (if any)
data = data.dropna(subset=feature_columns + [target_column])

# 2. Convert columns to numeric types
for col in feature_columns + [target_column]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 3. Drop rows that might have become NaN after conversion
data = data.dropna(subset=feature_columns + [target_column])

# (Optional) If you want to use categorical columns like Month, Decade, or Stage,
# you can encode them. For now, we'll work with the numeric columns only.

# Define features (X) and target (y)
X = data[feature_columns]
y = data[target_column]

# ------------------------------------------------------------------
# TRAIN/TEST SPLIT
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------
# TRAIN XGBOOST MODEL
# ------------------------------------------------------------------
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# EVALUATE THE MODEL
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nTest Mean Squared Error:", mse)

# ------------------------------------------------------------------
# PREDICTION EXAMPLE
# ------------------------------------------------------------------
def predict_irrigation(new_data_df):
    """
    Given a DataFrame with the same feature columns as the training data,
    predict future irrigation requirements.
    
    Parameters:
        new_data_df (pd.DataFrame): New data with columns in feature_columns.
        
    Returns:
        predictions (np.array): Predicted irrigation requirements (mm/dec).
    """
    # Convert features to numeric in new data as well
    for col in feature_columns:
        new_data_df[col] = pd.to_numeric(new_data_df[col], errors='coerce')
    return model.predict(new_data_df[feature_columns])

# Example: predict for the first 5 rows in X_test
new_data = X_test.head(5)
predictions = predict_irrigation(new_data)
print("\nPredicted Irrigation Requirements (mm/dec) for new data:")
print(predictions)

# ------------------------------------------------------------------
# SAVE THE MODEL (optional)
# ------------------------------------------------------------------
joblib.dump(model, "xgboost_irrigation_model.pkl")
print("\nModel saved as 'xgboost_irrigation_model.pkl'.")
