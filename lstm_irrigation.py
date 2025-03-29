import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- Load Data ---
base_path = os.path.dirname(__file__)  
csv_file_path = os.path.join(base_path, "climate_irrigation_data.csv")

data = pd.read_csv(csv_file_path)
data.columns = data.columns.str.strip()

# Select relevant features
feature_columns = ["Min Temp", "Max Temp", "Humidity", "Kc"]
target_column = "ETc"

# Convert to numeric & drop NaNs
data = data[feature_columns + [target_column]].apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

# --- Feature Engineering ---
# Create additional features
data["Temp_Diff"] = data["Max Temp"] - data["Min Temp"]
data["Humidity_Squared"] = data["Humidity"] ** 2

# Update feature list
feature_columns.extend(["Temp_Diff", "Humidity_Squared"])

# --- Normalize Features ---
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(data[feature_columns])
y_scaled = scaler_y.fit_transform(data[[target_column]])

joblib.dump(scaler_X, os.path.join(base_path, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(base_path, "scaler_y.pkl"))

# --- Create Sequences ---
def create_sequences(X, y, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 10  # You can experiment with different values
X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# --- Train-Test Split ---
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# --- Build Improved LSTM Model ---
model = Sequential([
    Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.3),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output Layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- Callbacks for Optimization ---
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- Train Model ---
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[lr_scheduler, early_stopping], verbose=1)

# --- Save Model ---
model_path = os.path.join(base_path, "improved_lstm_etc_model.h5")
model.save(model_path)
print(f"Model saved as {model_path}")

# --- Make Predictions ---
predictions = model.predict(X_test)

# Convert predictions & actual values back to original scale
predicted_ETc = scaler_y.inverse_transform(predictions)
actual_ETc = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# --- Plot Predictions vs Actual ---
plt.figure(figsize=(10, 6))
plt.plot(actual_ETc, label='Actual ETc', color='blue')
plt.plot(predicted_ETc, label='Predicted ETc', color='red', linestyle='dashed')
plt.title('ETc Prediction using Improved LSTM')
plt.xlabel('Time Steps')
plt.ylabel('ETc')
plt.legend()
plt.show()
