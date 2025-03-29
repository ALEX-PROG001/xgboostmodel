import numpy as np
import pandas as pd
import random
import os

# Load dataset
file_path = "climate_irrigation_data.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Ensure column names are stripped of extra spaces
df.columns = df.columns.str.strip()

# Select relevant columns
columns_to_use = ["Min Temp", "Max Temp", "Humidity", "Kc", "ETc"]
df = df[columns_to_use]

# Convert columns to numeric and drop missing values
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# Debugging: Print data info
print("Dataset loaded successfully!")
print(df.head())

# Define discretization (Binning)
num_bins = 5  # Number of bins
temp_bins = np.linspace(df["Min Temp"].min(), df["Max Temp"].max(), num=num_bins)
humidity_bins = np.linspace(df["Humidity"].min(), df["Humidity"].max(), num=num_bins)
kc_bins = np.linspace(df["Kc"].min(), df["Kc"].max(), num=num_bins)
etc_bins = np.linspace(df["ETc"].min(), df["ETc"].max(), num=num_bins)

# Categorize data into bins
df["Temp Category"] = np.digitize(df["Min Temp"], temp_bins, right=True)
df["Humidity Category"] = np.digitize(df["Humidity"], humidity_bins, right=True)
df["Kc Category"] = np.digitize(df["Kc"], kc_bins, right=True)
df["ETc Category"] = np.digitize(df["ETc"], etc_bins, right=True)

# Define state space
num_temp_states = len(temp_bins) + 1
num_humidity_states = len(humidity_bins) + 1
num_kc_states = len(kc_bins) + 1
state_size = (num_temp_states, num_humidity_states, num_kc_states)

# Define action space (ETc prediction categories)
num_actions = len(etc_bins) + 1
actions = list(range(num_actions))

# Initialize Q-table with zeros
Q_table = np.zeros(state_size + (num_actions,))

# Define Q-learning parameters
epsilon = 0.1  # Exploration rate
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 1000
max_steps = 10  # Max steps per episode

print("Starting Q-learning training...")


def train_q_learning():
    global epsilon, learning_rate  # Declare global inside the function before modifying

    for episode in range(num_episodes):
        # Initialize a random state
        state = (
            random.randint(0, num_temp_states - 1),
            random.randint(0, num_humidity_states - 1),
            random.randint(0, num_kc_states - 1),
        )

        steps = 0
        while steps < max_steps:
            # Choose action using ε-greedy policy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)  # Explore
            else:
                action = np.argmax(Q_table[state])  # Exploit

            # Sample a random row from dataset
            row = df.sample(n=1).iloc[0]
            actual_etc = row["ETc"]

            # Reward function (penalize deviation from actual ETc)
            reward = -abs(actual_etc - action)
            if abs(actual_etc - action) < 0.5:
                reward += 1  # Bonus for close prediction

            # Determine next state
            next_state = (
                np.digitize(row["Min Temp"], temp_bins, right=True),
                np.digitize(row["Humidity"], humidity_bins, right=True),
                np.digitize(row["Kc"], kc_bins, right=True),
            )

            # Q-learning update (Bellman equation)
            Q_table[state + (action,)] += learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state + (action,)]
            )

            # Move to next state
            state = next_state
            steps += 1

        # Decay epsilon and learning rate
        epsilon = max(0.01, epsilon * 0.99)
        learning_rate = 0.1 / (1 + 0.001 * episode)  # Gradual learning rate decay

        if episode % 100 == 0:
            np.save("q_table.npy", Q_table)
            print(f"Episode {episode}: Saved Q-table.")

    print("Training complete!")
    np.save("q_table.npy", Q_table)


# Run training
train_q_learning()

# Function to predict ETc
def predict_etc(min_temp, humidity, kc):
    state = (
        np.digitize(min_temp, temp_bins, right=True),
        np.digitize(humidity, humidity_bins, right=True),
        np.digitize(kc, kc_bins, right=True),
    )
    action = np.argmax(Q_table[state])  # Choose best ETc prediction
    return action


# Example usage
test_min_temp = 10.4
test_humidity = 68.0
test_kc = 0.4
predicted_etc = predict_etc(test_min_temp, test_humidity, test_kc)
print(f"\nPredicted ETc: {predicted_etc} mm/day")
