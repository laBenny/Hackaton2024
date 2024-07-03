import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'shuffled_train_bus_sche.csv'
data = pd.read_csv(file_path, encoding='cp1255')

# Define the relevant features and target
features = ['latitude', 'longitude', 'direction', 'arrival_is_estimated', 'mekadem_nipuach_luz', 'passengers_continue_menupach']
target = 'passengers_up'

# Extract features and target
X = data[features]
y = data[target]

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Convert predictions to non-negative integers
y_pred = y_pred.round().astype(int)
y_pred = [max(0, pred) for pred in y_pred]

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Prepare the output in the required format
output = pd.DataFrame({
    'trip_id_unique_station': data.loc[y_test.index, 'trip_id_unique_station'],
    'passengers_up': y_pred
})

# Save the output to a CSV file
output.to_csv('predicted_passengers_up.csv', index=False)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
