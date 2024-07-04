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
with open('IML.hackathon.2024-main/corrected_heb_table.csv', 'rb') as f:
    result = chardet.detect(f.read())
file_encoding = result['encoding']

file_path = 'IML.hackathon.2024-main/corrected_heb_table.csv'
data = pd.read_csv(file_path, encoding=file_encoding)

# Feature engineering
data['hour_of_departure'] = pd.to_datetime(data['arrival_time']).dt.hour
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
    'trip_duration': 'first'
}).reset_index()

# Convert categorical variables to numeric where possible
agg_data['alternative'] = pd.to_numeric(agg_data['alternative'], errors='coerce')
agg_data['cluster'] = pd.to_numeric(agg_data['cluster'], errors='coerce')
agg_data = agg_data.fillna(0)

# Visualize correlations
corr_features = agg_data.drop(columns=['trip_id_unique'])
corr_matrix = corr_features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Define features and target
features = [
    'line_id', 'direction', 'alternative', 'cluster', 'station_index',
    'passengers_up', 'passengers_continue', 'mekadem_nipuach_luz',
    'passengers_continue_menupach', 'hour_of_departure'
]
target = 'trip_duration'

agg_data_encoded = pd.get_dummies(agg_data[features])

X_train, X_test, y_train, y_test = train_test_split(agg_data_encoded, agg_data[target], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Random Forest finished training with Mean Squared Error: {mse}')

# Load the new test set
with open('IML.hackathon.2024-main\data\HU.BER\X_trip_duration.csv', 'rb') as f:
    result = chardet.detect(f.read())
test_file_encoding = result['encoding']

test_file_path = 'IML.hackathon.2024-main\data\HU.BER\X_trip_duration.csv'
test_data = pd.read_csv(test_file_path, encoding=test_file_encoding)

# Preprocess the test set
test_data['hour_of_departure'] = pd.to_datetime(test_data['arrival_time']).dt.hour
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
    'hour_of_departure': 'first'
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

output_file_path = 'trip_duration_predictions.csv'
output.to_csv(output_file_path, index=False)
print(f'Predictions saved to {output_file_path}')
