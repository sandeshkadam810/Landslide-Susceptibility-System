import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('EDAI3\dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)
y_probabilities = rf_classifier.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction using new data
new_data = pd.DataFrame({
    'slope': [10],
    'precipitation': [200],
    'elevation': [13],
    'soil_type': [1],
    'rainfall': [250]
})

prediction_probabilities = rf_classifier.predict_proba(new_data)
print("Prediction probabilities for new data:")
print(f"Probability of No Landslide: {prediction_probabilities[0][0]:.2f}")
print(f"Probability of Landslide: {prediction_probabilities[0][1]:.2f}")
