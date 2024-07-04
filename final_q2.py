import pandas as pd
import numpy as np
import chardet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the new training dataset with detected encoding
file_path = r'IML.hackathon.2024-main/data/HU.BER/train_bus_schedule.csv'
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
file_encoding = result['encoding']

data = pd.read_csv(file_path, encoding=file_encoding)

# Ensure columns are datetime
data['arrival_time'] = pd.to_datetime(data['arrival_time'])
data['door_closing_time'] = pd.to_datetime(data['door_closing_time'])

# Generate the trip duration in minutes, handling trips that span over midnight
data['trip_duration'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds() / 60
negative_duration_mask = data['trip_duration'] < 0
data.loc[negative_duration_mask, 'trip_duration'] += 24 * 60

# Feature engineering
data['hour_of_departure'] = data['arrival_time'].dt.hour
data['minute_of_departure'] = data['arrival_time'].dt.minute

# Aggregate data by trip
agg_data = data.groupby('trip_id_unique').agg({
    'line_id': 'first',
    'direction': 'first',
    'alternative': 'first',
    'cluster': 'first',
    'station_index': 'count',
    'passengers_up': 'sum',
    'passengers_continue': 'sum',
    'mekadem_nipuach_luz': 'mean',
    'passengers_continue_menupach': 'sum',
    'hour_of_departure': 'first',
    'minute_of_departure': 'first',
    'trip_duration': 'first'
}).reset_index()

# Convert categorical variables to numeric where possible
agg_data['alternative'] = pd.to_numeric(agg_data['alternative'], errors='coerce')
agg_data['cluster'] = pd.to_numeric(agg_data['cluster'], errors='coerce')
agg_data = agg_data.fillna(0)

# Define features and target
features = [
    'line_id', 'direction', 'alternative', 'cluster', 'station_index',
    'passengers_up', 'passengers_continue', 'mekadem_nipuach_luz',
    'passengers_continue_menupach', 'hour_of_departure', 'minute_of_departure'
]
target = 'trip_duration'

agg_data_encoded = pd.get_dummies(agg_data[features])

# Ensure that the split is done based on trip_id_unique to maintain trip integrity
unique_trip_ids = agg_data['trip_id_unique'].unique()
train_ids, test_ids = train_test_split(unique_trip_ids, test_size=0.2, random_state=42)

train_data = agg_data_encoded[agg_data['trip_id_unique'].isin(train_ids)]
test_data = agg_data_encoded[agg_data['trip_id_unique'].isin(test_ids)]

y_train = agg_data.loc[agg_data['trip_id_unique'].isin(train_ids), target]
y_test = agg_data.loc[agg_data['trip_id_unique'].isin(test_ids), target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test_scaled)
y_pred=y_pred/60

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Optimized Random Forest finished training with Mean Squared Error: {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Load the new test set
with open('IML.hackathon.2024-main\data\HU.BER\X_trip_duration.csv', 'rb') as f:
    result = chardet.detect(f.read())
test_file_encoding = result['encoding']

test_file_path = 'IML.hackathon.2024-main\data\HU.BER\X_trip_duration.csv'
test_data = pd.read_csv(test_file_path, encoding=test_file_encoding)

# Preprocess the test set
test_data['arrival_time'] = pd.to_datetime(test_data['arrival_time'])
test_data['hour_of_departure'] = test_data['arrival_time'].dt.hour
test_data['minute_of_departure'] = test_data['arrival_time'].dt.minute

test_agg_data = test_data.groupby('trip_id_unique').agg({
    'line_id': 'first',
    'direction': 'first',
    'alternative': 'first',
    'cluster': 'first',
    'station_index': 'count',
    'passengers_up': 'sum',
    'passengers_continue': 'sum',
    'mekadem_nipuach_luz': 'mean',
    'passengers_continue_menupach': 'sum',
    'hour_of_departure': 'first',
    'minute_of_departure': 'first'
}).reset_index()

test_agg_data['alternative'] = pd.to_numeric(test_agg_data['alternative'], errors='coerce')
test_agg_data['cluster'] = pd.to_numeric(test_agg_data['cluster'], errors='coerce')
test_agg_data = test_agg_data.fillna(0)

test_agg_data_encoded = pd.get_dummies(test_agg_data[features])
X_test_final = test_agg_data_encoded.reindex(columns=agg_data_encoded.columns, fill_value=0)
X_test_final_scaled = scaler.transform(X_test_final)

test_trip_duration_predictions = best_rf.predict(X_test_final_scaled)
test_trip_duration_predictions = np.maximum(test_trip_duration_predictions, 0)

output = pd.DataFrame({
    'trip_id_unique': test_agg_data['trip_id_unique'],
    'trip_duration_in_minutes': test_trip_duration_predictions
})

output_file_path = 'trip_duration_predictions_corrected.csv'
output.to_csv(output_file_path, index=False)
print(f'Predictions saved to {output_file_path}')

# Compute and print the squared loss and RMSE for training
squared_loss = np.mean((y_test - y_pred) ** 2)
print(f'Squared Loss: {squared_loss}')
print(f'Root Mean Squared Error (RMSE): {np.sqrt(squared_loss)}')
