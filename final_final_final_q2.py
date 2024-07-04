import pandas as pd
import numpy as np
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the training dataset with detected encoding
file_path = r'IML.hackathon.2024-main/data/HU.BER/train_bus_schedule.csv'
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
file_encoding = result['encoding']

data = pd.read_csv(file_path, encoding=file_encoding)

# Feature engineering
data['arrival_time'] = pd.to_datetime(data['arrival_time'])
data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], errors='coerce')
data['hour_of_departure'] = data['arrival_time'].dt.hour
data['minute_of_departure'] = data['arrival_time'].dt.minute

# Calculate trip_duration if missing
if 'trip_duration' not in data.columns or (data['trip_duration'] == 0).any():
    first_station = data.groupby('trip_id_unique').first().reset_index()
    last_station = data.groupby('trip_id_unique').last().reset_index()
    data = data.merge(first_station[['trip_id_unique', 'arrival_time']], on='trip_id_unique', suffixes=('', '_first'))
    data = data.merge(last_station[['trip_id_unique', 'arrival_time']], on='trip_id_unique', suffixes=('', '_last'))
    data['trip_duration'] = (data['arrival_time_last'] - data['arrival_time_first']).dt.total_seconds() / 60
    data.drop(columns=['arrival_time_first', 'arrival_time_last'], inplace=True)

# Correct negative and zero trip durations
data['trip_duration'] = np.where(data['trip_duration'] < 0, data['trip_duration'] + 1440, data['trip_duration'])
data['trip_duration'] = np.where(data['trip_duration'] == 0, 1, data['trip_duration'])

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

# Ensure the train and test sets are randomly split every time
X_train, X_test, y_train, y_test = train_test_split(agg_data_encoded, agg_data[target], test_size=0.2, random_state=None)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=None, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Random Forest finished training with Mean Squared Error: {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Load the new test set
test_file_path = 'IML.hackathon.2024-main/data/HU.BER/X_trip_duration.csv'
with open(test_file_path, 'rb') as f:
    result = chardet.detect(f.read())
test_file_encoding = result['encoding']

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

test_trip_duration_predictions = rf.predict(X_test_final_scaled)
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
