import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('EDAI3/dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(probability=True)  # Enabling probability estimates

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Predict probability estimates on the test data
y_probabilities = svm_classifier.predict_proba(X_test)  # Probabilities for both classes

# Calculate accuracy on the test set
y_predictions = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)
print(f"Accuracy on Test Set: {accuracy:.2f}")

# Example prediction using new data
new_data = pd.DataFrame({
    'slope': [10],
    'precipitation': [200],
    'elevation': [13],
    'soil_type': [1],
    'rainfall': [250]
})

prediction_probabilities = svm_classifier.predict_proba(new_data)
probability_landslide = prediction_probabilities[0, 1]
probability_no_landslide = prediction_probabilities[0, 0]

print(f"Probability of Landslide: {probability_landslide:.2f}")
print(f"Probability of No Landslide: {probability_no_landslide:.2f}")
