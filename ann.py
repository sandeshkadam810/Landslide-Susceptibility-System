import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a simple feedforward neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # 2 output nodes for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict probabilities on the test data
y_probabilities = model.predict(X_test_scaled)

# Convert probabilities to class predictions
y_pred = y_probabilities.argmax(axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction using new data
new_data = pd.DataFrame({
    'slope': [14],
    'precipitation': [7],
    'elevation': [53],
    'soil_type': [2],
    'rainfall': [22]
})
new_data_scaled = scaler.transform(new_data)
prediction_probabilities = model.predict(new_data_scaled)
print("Prediction probabilities for new data:")
print(f"Probability of No Landslide: {prediction_probabilities[0][0]:.2f}")
print(f"Probability of Landslide: {prediction_probabilities[0][1]:.2f}")
