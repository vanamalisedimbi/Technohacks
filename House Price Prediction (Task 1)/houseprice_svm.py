import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Get the path to the CSV file from the user
csv_path = input("Enter the path to the CSV file: ")

# Load the data from the CSV file
data = pd.read_csv(csv_path)

# Convert 'yes' and 'no' to binary values (0 and 1)
data.replace({'yes': 1, 'no': 0}, inplace=True)

# Separate features and target variable
X = data.iloc[:, 1:]  # Features (excluding 'price')
y = data["price"]  # Target variable

# Specify categorical columns for one-hot encoding
categorical_columns = ['furnishingstatus']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_columns)],
    remainder='passthrough'
)

# Apply preprocessing to the categorical columns
X_preprocessed = preprocessor.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_preprocessed)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Support Vector Machine model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predictions vs Actual Values")
plt.show()
