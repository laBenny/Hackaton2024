import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic

FEATURES_TO_DROP = ["trip_id_unique_station", "trip_id_unique", "station_name", "trip_id"]
CATEGORIAL_FEATURES = ["part", "line_id", "alternative", "station_id", "station_index", "cluster"]
STATE = 30
TRAIN_PERCENTAGE = 75
OUTPUT_PATH = "."

def drop_bad_features(X: pd.DataFrame) -> pd.DataFrame:
    """ deletes FEATURES_TO_DROP from dataframe
    """
    for feature in FEATURES_TO_DROP:
        X = X.drop(feature, axis=1)
    return X

def process_categorials(X: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(X, columns=CATEGORIAL_FEATURES, drop_first=False)
    return X

def general_preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """ adds new features to the dataframe, to be used on the train and test set
    """
    X = drop_bad_features(X)
    X = process_categorials(X)

    X['arrival_time'] = pd.to_datetime(X['arrival_time'], errors='coerce')
    X = X.dropna(subset=['arrival_time'])
    X['arrival_hour'] = X['arrival_time'].dt.hour
    X = X.drop('arrival_time', axis=1)
    X['arrival_hour^2'] = X['arrival_hour']**2
    X['arrival_hour^3'] = X['arrival_hour']**3

    X["arrival_is_estimated"] = X["arrival_is_estimated"].map({'TRUE': 1, 'FALSE': 0}).fillna(0)

    # handle lat and long

    return X


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # concatenate X and y to keep their rows corresponding
    X_y = pd.concat([X, y], axis=1)
    X_y = X_y.dropna().drop_duplicates()

    # drop unused features and add new features, same is done to test set
    X_y = general_preprocess(X_y)



    #v re-devide X and y
    X: pd.DataFrame = X_y.drop("passengers_up", axis=1)
    y: pd.Series = X_y.passengers_up
    return X, y

   
def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # drop unused features and add new features, same is done to train set
    X = general_preprocess(X)

    # # replace invalid or empty cells with the median of its column
    # for feature in X:
    #     median = X[feature].median()
    #     X[feature].fillna(median, inplace=True)

    return X


if __name__ == '__main__':
    # read csv into dataframe
    df = pd.read_csv("C:\\Users\\amosd\\OneDrive\\שולחן העבודה\\hackaton\\train_bus_schedule.csv", encoding='cp1252')
    if df.empty:
        raise ValueError("DataFrame is empty")
    X: pd.DataFrame = df.drop("passengers_up", axis=1)
    y: pd.Series = df.passengers_up

    partial_X, rest_of_X, partial_y, rest_of_y = train_test_split(X, y, train_size=0.05, random_state=STATE)

    # Question 2 - split train test
    train_X, test_X, train_y, test_y = train_test_split(partial_X, partial_y, train_size=TRAIN_PERCENTAGE/100, random_state=STATE)

    # Question 3 - preprocessing of housing prices train dataset
    train_X, train_y = preprocess_train(train_X, train_y)

    # Question 5 - preprocess the test data
    test_X = preprocess_test(test_X)

    train_X.to_csv(os.path.join(OUTPUT_PATH, 'train_X.csv'), index=False)

