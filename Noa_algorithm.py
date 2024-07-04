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

# Convert all columns except categorical ones to numeric where possible
categorical_columns = ['part', 'trip_id_unique', 'cluster', 'station_name', 'arrival_time', 'door_closing_time', 'trip_id_unique_station']
data[categorical_columns] = data[categorical_columns].astype(str)
data_numeric = data.drop(columns=categorical_columns).apply(pd.to_numeric, errors='coerce', downcast='float')

# Include categorical columns back into the dataframe
data = pd.concat([data_numeric, data[categorical_columns]], axis=1)

# Create new features
data['latitude_longitude'] = data['latitude'] * data['longitude']
data['arrival_station_product'] = data['arrival_is_estimated'] * data['station_index']

# Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Handle missing values, removing features with all NaN values
X = X.dropna(axis=1, how='all')

# Handle missing values for numeric columns
numeric_columns = X.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
X_imputed_numeric = imputer.fit_transform(X[numeric_columns])
X_imputed_numeric = pd.DataFrame(X_imputed_numeric, columns=numeric_columns)

# Include categorical columns back after imputation
X_imputed = pd.concat([X_imputed_numeric, X[categorical_columns].reset_index(drop=True)], axis=1)

y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Normalize numerical features
scaler = StandardScaler()
X_scaled_numeric = scaler.fit_transform(X_imputed[numeric_columns])
X_scaled_numeric = pd.DataFrame(X_scaled_numeric, columns=numeric_columns)

# Include categorical columns back after scaling
X_scaled = pd.concat([X_scaled_numeric, X_imputed[categorical_columns].reset_index(drop=True)], axis=1)

# Ensure that the column names are preserved after scaling
feature_names = X_scaled.columns

# Define and train the Random Forest model with best hyperparameters
best_params = {
    'bootstrap': True,
    'max_depth': 18,
    'min_samples_leaf': 2,
    'min_samples_split': 6,
    'n_estimators': 117
}

# Iteratively use 5%, 20%, 50%, and 75% of the database
percentages = [0.05, 0.2, 0.5, 0.75]

results = []

for perc in percentages:
    # Sample the specified percentage of the data
    data_sampled = data.sample(frac=perc, random_state=42)
    
    # Separate the sampled data into features and target
    X_sampled = data_sampled.drop(columns=[target])
    y_sampled = data_sampled[target]
    
    # Handle missing values for the sampled data
    X_sampled_imputed_numeric = imputer.fit_transform(X_sampled[numeric_columns])
    X_sampled_imputed_numeric = pd.DataFrame(X_sampled_imputed_numeric, columns=numeric_columns)
    X_sampled_imputed = pd.concat([X_sampled_imputed_numeric, X_sampled[categorical_columns].reset_index(drop=True)], axis=1)

    y_sampled_imputed = imputer.fit_transform(y_sampled.values.reshape(-1, 1)).ravel()
    
    # Normalize numerical features for the sampled data
    X_sampled_scaled_numeric = scaler.fit_transform(X_sampled_imputed[numeric_columns])
    X_sampled_scaled_numeric = pd.DataFrame(X_sampled_scaled_numeric, columns=numeric_columns)
    X_sampled_scaled = pd.concat([X_sampled_scaled_numeric, X_sampled_imputed[categorical_columns].reset_index(drop=True)], axis=1)
    
    # Split the sampled data into training and testing sets
    random_seed = np.random.randint(10000)  # Change this to any random number for each run
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X_sampled_scaled, y_sampled_imputed, data_sampled.index, test_size=0.2, random_state=random_seed)

    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    model.fit(X_train[numeric_columns], y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test[numeric_columns])
    y_pred = y_pred.round().astype(int)
    y_pred = np.maximum(0, y_pred)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'Percentage: {perc*100}% - Random Forest - MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    # Feature importance
    feature_importances = pd.DataFrame({'Feature': numeric_columns, 'Importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    # Prepare the final output
    y_final_pred = model.predict(X_test[numeric_columns])
    y_final_pred = y_final_pred.round().astype(int)
    y_final_pred = np.maximum(0, y_final_pred)

    output = pd.DataFrame({
        'trip_id_unique_station': X_test['trip_id_unique_station'],
        'passengers_up': y_final_pred
    })

    # Save the output to a CSV file
    output_filename = f'predicted_passengers_up_random_forest_{int(perc*100)}.csv'
    output.to_csv(output_filename, index=False)

    # Save the results
    results.append({
        'Percentage': perc,
        'MAE': mae,
        'RMSE': rmse,
        'Feature Importances': feature_importances,
        'Output Filename': output_filename
    })

    # Display the feature importance plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title(f'Feature Importances from Random Forest ({int(perc*100)}% of Data)')
    plt.show()

# Display results summary
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df)
