import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('animals.csv')

# Select features and target variable
X = df[['continent_encoded', 'habitat_encoded', 'food_encoded', 'population', 'temperature', 'survival_rate']]
y = df['conservation_status_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
dtc = DecisionTreeClassifier()

# Train the classifier on the training data
dtc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dtc.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get top 3 animals in each conservation status
def get_top_animals(status_encoded):
    top_animals = df[df['conservation_status_encoded'] == status_encoded].sort_values(by='population', ascending=False).head(3)
    return top_animals[['Common name', 'Scientific name', 'Population']]

print("\nTop 3 Critically Endangered Animals:")
print(get_top_animals(0))

print("\nTop 3 Endangered Animals:")
print(get_top_animals(1))

print("\nTop 3 Vulnerable Animals:")
print(get_top_animals(2))

print("\nTop 3 Near Threatened Animals:")
print(get_top_animals(3))

print("\nTop 3 Least Concern Animals:")
print(get_top_animals(4))