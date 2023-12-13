from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your dataset
df = read_csv('../Predic/static/dataset_recent/insurance.csv')

# Separate features (X) and target variable (y)
X = df[['age', 'sex', 'bmi', 'children', 'smoker']]
Y = df['charges']

# Set the number of folds for K-Fold Cross Validation
num_folds = 10
mse_scores = []

# Create a Linear Regression model
model = LinearRegression()

# Initialize the K-Fold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-Fold Cross Validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Use iloc to index DataFrame by position
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Train the model on the training data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE) for this fold
    mse = mean_squared_error(Y_test, Y_pred)
    mse_scores.append(mse)

# Calculate the mean MSE across all folds
mean_mse = np.mean(mse_scores)

# Display the MSE for each fold
for fold, mse in enumerate(mse_scores, start=1):
    print(f"Fold {fold} - Mean Squared Error (MSE): {mse:.3f}")

# Display the mean MSE across all folds
print(f"Mean MSE across all folds: {mean_mse:.3f}")
