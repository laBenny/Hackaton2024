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

# Preserve trip_id_unique_station as string
trip_id_unique_station = data['trip_id_unique_station']

# Define the target
target = 'passengers_up'

# Convert all columns except trip_id_unique_station to numeric where possible
data = data.apply(pd.to_numeric, errors='coerce', downcast='float')
data['trip_id_unique_station'] = trip_id_unique_station

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
X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))
X_imputed = pd.DataFrame(X_imputed, columns=X.select_dtypes(include=[np.number]).columns)
X_imputed['trip_id_unique_station'] = trip_id_unique_station

y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed.select_dtypes(include=[np.number]))
X_scaled = pd.DataFrame(X_scaled, columns=X_imputed.select_dtypes(include=[np.number]).columns)
X_scaled['trip_id_unique_station'] = trip_id_unique_station

# Ensure that the column names are preserved after scaling
feature_names = X_scaled.columns

# Split the data into training and testing sets with a different random seed each time
random_seed = np.random.randint(10000)  # Change this to any random number for each run
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y_imputed, data.index, test_size=0.2, random_state=random_seed)

# Define and train the Random Forest model with best hyperparameters
best_params = {
    'bootstrap': True,
    'max_depth': 18,
    'min_samples_leaf': 2,
    'min_samples_split': 6,
    'n_estimators': 117
}

model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    bootstrap=best_params['bootstrap'],
    random_state=42
)
model.fit(X_train.drop(columns=['trip_id_unique_station']), y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test.drop(columns=['trip_id_unique_station']))
y_pred = y_pred.round().astype(int)
y_pred = np.maximum(0, y_pred)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Random Forest - MAE: {mae:.4f}, RMSE: {rmse:.4f}')

# Feature importance
feature_importances = pd.DataFrame({'Feature': feature_names.drop('trip_id_unique_station'), 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Prepare the final output
y_final_pred = model.predict(X_test.drop(columns=['trip_id_unique_station']))
y_final_pred = y_final_pred.round().astype(int)
y_final_pred = np.maximum(0, y_final_pred)

output = pd.DataFrame({
    'trip_id_unique_station': X_test['trip_id_unique_station'],
    'passengers_up': y_final_pred
})

# Save the output to a CSV file
output.to_csv(f'predicted_passengers_up_best_random_forest.csv', index=False)

# Display the feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Random Forest')
plt.show()
