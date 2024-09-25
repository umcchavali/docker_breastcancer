from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Read the CSV
df = pd.read_csv('/mnt/datalake/delta/breastcancer.csv')
#df = pd.read_csv('breastcancer.csv')


# Getting dummies for categorical variables
df = pd.get_dummies(df,columns=['menopause', 'breast', 'breast_quad', 'node_caps', 'irradiat'])

# Creaing a copy of df so that thhe original dataset remains intact
df1= df.copy()

X = df1.drop('target', axis=1)
y = df1['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
import joblib
joblib.dump(model, '/mnt/datalake/delta/uma_breastcancer_model.pkl')