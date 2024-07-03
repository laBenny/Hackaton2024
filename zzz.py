import pandas as pd
import numpy as np
import chardet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Create new features
data['latitude_longitude'] = data['latitude'] * data['longitude']
data['arrival_station_product'] = data['arrival_is_estimated'] * data['station_index']

# Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Handle missing values, removing features with all NaN values
X = X.dropna(axis=1, how='all')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Ensure that the column names are preserved after scaling
feature_names = X.columns

# Split the data into training and testing sets while keeping track of indices
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X_scaled, y_imputed, data.index, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)
y_pred = y_pred.round().astype(int)
y_pred = np.maximum(0, y_pred)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Random Forest - MAE: {mae:.4f}, RMSE: {rmse:.4f}')

# Feature importance
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Prepare the final output
y_final_pred = model.predict(X_test)
y_final_pred = y_final_pred.round().astype(int)
y_final_pred = np.maximum(0, y_final_pred)

output = pd.DataFrame({
    'trip_id_unique_station': data.loc[test_indices, 'trip_id_unique_station'],
    'passengers_up': y_final_pred
})

# Save the output to a CSV file
output.to_csv(f'predicted_passengers_up_random_forest.csv', index=False)

# Display the feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Random Forest')
plt.show()
