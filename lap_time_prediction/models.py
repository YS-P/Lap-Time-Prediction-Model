# models.py
from typing import Dict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from config import RANDOM_STATE

def train_global_model(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_global(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return pred, mae

def mae_per_driver(test_df, y_test, pred) -> Dict[str, float]:
    drivers = test_df['Driver'].unique()
    result = {}
    for drv in drivers:
        idx = (test_df['Driver'] == drv)
        y_true = y_test[idx]
        y_pred = pred[idx]
        if len(y_true) == 0:
            continue
        result[drv] = mean_absolute_error(y_true, y_pred)
    return result

def train_driver_specific_models(train_df, X_train, y_train, test_df, X_test):
    drivers = train_df['Driver'].unique()
    models = {}
    pred_all = np.zeros(len(test_df))

    for drv in drivers:
        train_idx = (train_df['Driver'] == drv)
        test_idx = (test_df['Driver'] == drv)

        if test_idx.sum() == 0 or train_idx.sum() < 5:
            continue

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train[train_idx], y_train[train_idx])
        pred_all[test_idx] = model.predict(X_test[test_idx])
        models[drv] = model

    return models, pred_all
