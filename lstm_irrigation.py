import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # To save the scaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Load Data ---
file_path = r"C:\Users\USER\Documents\irrigation model\irrigation model\climate_irrigation_data.csv"

# Read CSV and clean column names
data = pd.read_csv(file_path, encoding='latin1')
data.columns = data.columns.str.strip()  # Remove hidden spaces

# Rename columns for easier reference
data.rename(columns={
    'Irr. Req. (mm/dec)': 'IrrReq',
    'ETc (mm/day)': 'ETc_day',
    'ETc (mm/dec)': 'ETc_dec',
    'Eff rain (mm/dec)': 'EffRain'
}, inplace=True)

# Drop categorical columns
data.drop(columns=['Month', 'Decade', 'Stage'], axis=1, errors='ignore', inplace=True)

# Convert all numeric columns to float
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert non-numeric to NaN

# Handle missing values
data.fillna(method='ffill', inplace=True)  # Forward-fill missing values

# Ensure 'IrrReq' is the first column
cols = data.columns.tolist()
cols.remove('IrrReq')
data = data[['IrrReq'] + cols]

# --- Normalize Features ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# --- Create Time Series Sequences ---
def create_sequences(dataset, time_steps=10):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i+time_steps])
        y.append(dataset[i+time_steps, 0])  # Target is 'IrrReq'
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(scaled_data, time_steps)

# --- Train-Test Split ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- Build LSTM Model ---
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- Train Model ---
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# --- Save Model ---
model.save('lstm_irrigation.h5')
print("Model saved as lstm_irrigation.h5")

# --- Make Predictions ---
predictions = model.predict(X_test)

# Function to reverse scaling for target variable
def inverse_transform_target(scaled_preds, scaler, n_features):
    temp = np.zeros((scaled_preds.shape[0], n_features))
    temp[:, 0] = scaled_preds[:, 0]  # Target is in the first column
    return scaler.inverse_transform(temp)[:, 0]

# Convert predictions & actual values back to original scale
n_features = data.shape[1]
predicted_irrigation = inverse_transform_target(predictions, scaler, n_features)
actual_irrigation = inverse_transform_target(y_test.reshape(-1, 1), scaler, n_features)

# --- Plot Predictions vs Actual Data ---
plt.figure(figsize=(10, 6))
plt.plot(actual_irrigation, label='Actual Irrigation Requirement', color='blue')
plt.plot(predicted_irrigation, label='Predicted Irrigation Requirement', color='red', linestyle='dashed')
plt.title('Irrigation Requirement Prediction')
plt.xlabel('Time Steps (each = 10 days)')
plt.ylabel('Irrigation Requirement (mm/dec)')
plt.legend()
plt.show()
