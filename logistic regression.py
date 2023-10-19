import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression classifier
log_reg = LogisticRegression()

# Train the classifier on the training data
log_reg.fit(X_train, y_train)

# Predict on the test data
y_pred = log_reg.predict(X_test)
y_probabilities = log_reg.predict_proba(X_test)

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

prediction_probabilities = log_reg.predict_proba(new_data)
print("Prediction probabilities for new data:")
print(f"Probability of No Landslide: {prediction_probabilities[0][0]:.2f}")
print(f"Probability of Landslide: {prediction_probabilities[0][1]:.2f}")
