import pandas as pd
import numpy as np
import chardet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Detect file encoding
with open('heb_table.csv', 'rb') as f:
    result = chardet.detect(f.read())
file_encoding = result['encoding']

# Load the dataset with detected encoding
file_path = 'heb_table.csv'
data = pd.read_csv(file_path, encoding=file_encoding)

# Define the target
target = 'passengers_up'

# Convert all columns to numeric where possible
data = data.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Select features based on correlation with the target
correlation_threshold = 0.1
features = correlation_matrix.index[abs(correlation_matrix[target]) > correlation_threshold].tolist()
features.remove(target)

# Extract features and target
X = data[features]
y = data[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets while keeping track of indices
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X_scaled, y_imputed, data.index, test_size=0.2, random_state=42)

# Define the models with some parameters to ensure efficiency
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
}

# Train and evaluate each model
best_model = None
best_mae = float('inf')
results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = y_pred.round().astype(int)
    y_pred = np.maximum(0, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append((model_name, mae, rmse))
    if mae < best_mae:
        best_mae = mae
        best_model = model_name

# Prepare the final output
best_model_instance = models[best_model]
best_model_instance.fit(X_train, y_train)
y_final_pred = best_model_instance.predict(X_test)
y_final_pred = y_final_pred.round().astype(int)
y_final_pred = np.maximum(0, y_final_pred)

# Use the correct indices from the original DataFrame for the output
output = pd.DataFrame({
    'trip_id_unique_station': data.loc[test_indices, 'trip_id_unique_station'],
    'passengers_up': y_final_pred
})

# Save the output to a CSV file
output.to_csv(f'predicted_passengers_up_{best_model.lower().replace(" ", "_")}.csv', index=False)

# Print the evaluation metrics and the best model
for result in results:
    print(f'{result[0]} - MAE: {result[1]:.4f}, RMSE: {result[2]:.4f}')
print(f'\nBest model: {best_model} with MAE: {best_mae:.4f}')

# Display the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
