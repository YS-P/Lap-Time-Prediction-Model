# features.py
import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE

def create_base_data(merged: pd.DataFrame) -> pd.DataFrame:
    data = merged.copy()
    data['LapTimeSec'] = data['LapTime'].dt.total_seconds()
    data['PrevLapTimeSec'] = data.groupby('Driver')['LapTimeSec'].shift(1)
    data = data.drop(columns='LapTime')
    data = data.dropna(subset=['PrevLapTimeSec'])
    return data

def train_test_split_data(data: pd.DataFrame):
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        shuffle=False,
        random_state=RANDOM_STATE
    )
    return train, test

def select_features(train: pd.DataFrame):
    numeric_cols = train.select_dtypes(include='number').columns
    corr = train[numeric_cols].corr()

    target_corr = corr['LapTimeSec'].drop('LapTimeSec')
    top8 = target_corr.abs().sort_values(ascending=False).head(8)

    selected_features = list(top8.index) + ['Compound', 'TrackStatus']
    return selected_features, corr, top8

def encode_train_test(train: pd.DataFrame, test: pd.DataFrame, selected_features):
    train_encoded = pd.get_dummies(train[selected_features], drop_first=True)
    test_encoded = pd.get_dummies(test[selected_features], drop_first=True)

    # match columns
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    X_train = train_encoded
    y_train = train['LapTimeSec']
    X_test = test_encoded
    y_test = test['LapTimeSec']

    return X_train, y_train, X_test, y_test
