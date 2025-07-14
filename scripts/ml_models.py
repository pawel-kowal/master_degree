import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def prepare_ml_data(df, target_col, lag_range):
    df_ml = df.copy()
    for lag in range(1, lag_range + 1):
        df_ml[f'{target_col}_lag{lag}'] = df_ml[target_col].shift(lag)
        for col in df.columns:
            if col != target_col:
                df_ml[f'{col}_lag{lag}'] = df_ml[col].shift(lag)
    df_ml['month'] = df_ml.index.month
    df_ml['quarter'] = df_ml.index.quarter
    df_ml['year'] = df_ml.index.year
    df_ml.dropna(inplace=True)
    X = df_ml.drop(target_col, axis=1)
    y = df_ml[target_col]
    return X, y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_xgb_model(X_train, y_train, params=None):
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def tune_xgb_model(X_train, y_train, param_grid, cv_splits=5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror'),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
