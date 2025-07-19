import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def prepare_ml_data(df, target_col, lag_range):
    df_ml = df.copy()

    # Tworzenie różnicowań, jeśli nie istnieją
    if 'stopa_bezrobocia_diff' not in df_ml.columns:
        df_ml['stopa_bezrobocia_diff'] = df_ml['stopa_bezrobocia'].diff()
    if 'stopa_bezrobocia_diff2' not in df_ml.columns:
        df_ml['stopa_bezrobocia_diff2'] = df_ml['stopa_bezrobocia_diff'].diff()

    # 2. Zapisz listę kolumn do lagowania (po dodaniu diffów)
    orig_cols = df_ml.columns.tolist()

    # 3. Laguj każdą z tych kolumn dla lag = 1…lag_range
    for col in orig_cols:
        df_ml[f'{col}_lag{lag_range}'] = df_ml[col].shift(lag_range)

    # Sezonowe cechy czasowe
    df_ml['month_sin'] = np.sin(2 * np.pi * df_ml.index.month / 12)
    df_ml['month_cos'] = np.cos(2 * np.pi * df_ml.index.month / 12)
    df_ml['quarter_sin'] = np.sin(2 * np.pi * df_ml.index.quarter / 4)
    df_ml['quarter_cos'] = np.cos(2 * np.pi * df_ml.index.quarter / 4)
    df_ml['year'] = df_ml.index.year

    # Usuń NA z powodu lagów i diffów
    df_ml.dropna(inplace=True)

    # Budujemy X tylko z:
    # • wszystkich oryginalnych zmiennych (bez target_col i pierwszej różnicy)
    # • oraz lagów target_col
    drop_cols = [target_col, 'stopa_bezrobocia', 'stopa_bezrobocia_diff']
    X = df_ml.drop(columns=drop_cols, errors='ignore')
    y = df_ml[target_col]
    return X, y

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_xgb_model(X_train, y_train, n_trials=50, use_optuna=False, params=None):
    if not use_optuna:
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

    # Optuna tuning
    def objective(trial):
        param_grid = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, valid_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            model = XGBRegressor(**param_grid)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = mean_squared_error(y_val, preds) ** 0.5
            scores.append(rmse)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    best_params['random_state'] = 42

    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    return best_model

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
