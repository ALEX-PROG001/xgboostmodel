import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load CSV file (assuming it's in the same folder as the script)
csv_file_path = os.path.join(os.path.dirname(__file__), "climate_irrigation_data.csv")

# Define features and target
feature_columns = ["Min Temp", "Max Temp", "Humidity", "Kc"]
target_column = "ETc"

# Load dataset
data = pd.read_csv(csv_file_path)

# Drop missing values and ensure numeric conversion
data = data.dropna(subset=feature_columns + [target_column])
data[feature_columns + [target_column]] = data[feature_columns + [target_column]].apply(pd.to_numeric, errors='coerce')
data = data.dropna()  # Drop rows that became NaN after conversion

# Define features and target
X = data[feature_columns]
y = data[target_column]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    cv=3, 
    scoring='neg_mean_squared_error', 
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Retrieve best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate on test data
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display results
print("\nBest Hyperparameters:", best_params)
print("\nTest Mean Squared Error after tuning:", mse)

# Save the best model
model_path = os.path.join(os.path.dirname(__file__), "xgboost_irrigation_model.pkl")
joblib.dump(best_model, model_path)
print(f"\nOptimized model saved as '{model_path}'.")
