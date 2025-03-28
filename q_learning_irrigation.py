import numpy as np
import pandas as pd
import random
import os
import sys

# Load dataset
file_path = r"C:\Users\USER\Documents\irrigation model\irrigation model\climate_irrigation_data.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Strip column names to remove extra spaces
df.columns = df.columns.str.strip()

# Select relevant columns
columns_to_use = ["Min Temp", "Max Temp", "Humidity", "Kc", "ETc (mm/dec)", "ETo", "Irr. Req. (mm/dec)"]
df = df[columns_to_use]

# Convert relevant columns to numeric values
for col in columns_to_use:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Debugging: Print data types and sample data
print("Data Types:\n", df.dtypes, flush=True)
print("\nSample Data:\n", df.head(), flush=True)

# Discretization (Binning)
num_bins = 5  # Number of bins
temp_bins = np.linspace(df["Min Temp"].min(), df["Max Temp"].max(), num=num_bins)
humidity_bins = np.linspace(df["Humidity"].min(), df["Humidity"].max(), num=num_bins)
kc_bins = np.linspace(df["Kc"].min(), df["Kc"].max(), num=num_bins)
irrigation_bins = np.linspace(df["Irr. Req. (mm/dec)"].min(), df["Irr. Req. (mm/dec)"].max(), num=num_bins)

# Assign categories using np.digitize (right=True ensures proper binning)
df["Temp Category"] = np.digitize(df["Min Temp"], temp_bins, right=True)
df["Humidity Category"] = np.digitize(df["Humidity"], humidity_bins, right=True)
df["Kc Category"] = np.digitize(df["Kc"], kc_bins, right=True)
df["Irrigation Category"] = np.digitize(df["Irr. Req. (mm/dec)"], irrigation_bins, right=True)

# Define state space (using Temp, Humidity, Kc categories)
num_temp_states = len(temp_bins) + 1
num_humidity_states = len(humidity_bins) + 1
num_kc_states = len(kc_bins) + 1
state_size = (num_temp_states, num_humidity_states, num_kc_states)

# Define action space (Irrigation levels)
num_actions = len(irrigation_bins) + 1
actions = list(range(num_actions))

# Initialize Q-table with zeros
Q_table = np.zeros(state_size + (num_actions,))

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration rate
num_episodes = 1000
max_steps = 10  # Maximum steps per episode

print("Starting Q-learning training...", flush=True)

# Training loop
for episode in range(num_episodes):
    # Initialize a random state (tuple of indices)
    state = (
        random.randint(0, num_temp_states - 1),
        random.randint(0, num_humidity_states - 1),
        random.randint(0, num_kc_states - 1),
    )
    
    steps = 0
    done = False
    while not done and steps < max_steps:
        # Choose action using ε-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action = np.argmax(Q_table[state])  # Exploit
        
        # Simulate reward: use a random row from the dataset as environment feedback
        row = df.sample(n=1).iloc[0]
        eto = row["ETo"]
        etc = row["ETc (mm/dec)"]
        irrigation_req = row["Irr. Req. (mm/dec)"]
        
        # Reward: penalize deviation from required irrigation and water wastage
        reward = -abs(irrigation_req - action) - (1 if eto < etc else 0)
        
        # Determine next state from the current row's values
        next_state = (
            np.digitize(row["Min Temp"], temp_bins, right=True),
            np.digitize(row["Humidity"], humidity_bins, right=True),
            np.digitize(row["Kc"], kc_bins, right=True),
        )
        
        # Q-learning update (Bellman equation)
        Q_table[state + (action,)] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state + (action,)]
        )
        
        # Transition to next state and increment step counter
        state = next_state
        steps += 1
        
        # Optional: If you want to end the episode based on some condition, set done=True here.
    
    # Print progress after each episode
    print(f"Episode {episode} completed.", flush=True)

print("Training complete!", flush=True)

# Save Q-table
q_table_path = r"C:\Users\USER\Documents\irrigation model\q_table.npy"
np.save(q_table_path, Q_table)
print(f"\nQ-learning model trained and saved at {q_table_path}", flush=True)

# Function to recommend irrigation based on input conditions
def recommend_irrigation(min_temp, humidity, kc):
    state = (
        np.digitize(min_temp, temp_bins, right=True),
        np.digitize(humidity, humidity_bins, right=True),
        np.digitize(kc, kc_bins, right=True),
    )
    action = np.argmax(Q_table[state])  # Choose best irrigation level
    return action

# Example usage of the recommendation function
test_min_temp = 5.0
test_humidity = 25.0
test_kc = 0.8
irrigation_recommendation = recommend_irrigation(test_min_temp, test_humidity, test_kc)
print(f"\nRecommended irrigation level: {irrigation_recommendation} mm/dec", flush=True)
