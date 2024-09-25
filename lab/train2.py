from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # For classification accuracy
import pandas as pd

# Read the CSV
df = pd.read_csv('/mnt/datalake/delta/breastcancer.csv')
#df = pd.read_csv('breastcancer.csv')

# Getting dummies for categorical variables
df = pd.get_dummies(df, columns=['menopause', 'breast', 'breast_quad', 'node_caps', 'irradiat'])

# Creaing a copy of df so that the original dataset remains intact
df2 = df.copy()

X = df2.drop('target', axis=1)
y = df2['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create and train the logistic regression model
model = LogisticRegression()  # Instantiate the logistic regression model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy for classification
print(f'Accuracy: {accuracy}')

# Save the model
import joblib
joblib.dump(model, '/mnt/datalake/delta/uma_breastcancer_model2.pkl')