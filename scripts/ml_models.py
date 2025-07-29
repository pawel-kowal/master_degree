import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import early_stopping

def prepare_ml_data(df, target_col):
    df_ml = df.copy()
    df_ml['target_diff'] = df_ml[target_col].diff()

    orig_cols = [col for col in df.columns if col != target_col]

    for lag in [1, 2, 3, 6, 12]:
        for col in orig_cols:
            df_ml[f'{col}_lag_{lag}'] = df_ml[col].shift(lag)

    df_ml['month_sin'] = np.sin(2 * np.pi * df_ml.index.month / 12)
    df_ml['month_cos'] = np.cos(2 * np.pi * df_ml.index.month / 12)
    df_ml['quarter_sin'] = np.sin(2 * np.pi * df_ml.index.quarter / 4)
    df_ml['quarter_cos'] = np.cos(2 * np.pi * df_ml.index.quarter / 4)
    df_ml['year'] = df_ml.index.year

    df_ml.dropna(inplace=True)

    drop_cols = [target_col, 'target_diff']
    X = df_ml.drop(columns=drop_cols, errors='ignore')
    y = df_ml['target_diff']
    # y = df_ml[target_col]

    return X, y

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

def train_lgbm_model(X_train, y_train, params=None, random_state=42):
    params = params or {}
    params.setdefault('random_state', random_state)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

def optimize_lgbm(X_tr, y_tr, n_trials=50, random_state=42):
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    import numpy as np

    def objective(trial):
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', -0.1, 0.1),
            'verbosity': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []

        for train_idx, val_idx in tscv.split(X_tr):
            X_train, X_val = X_tr[train_idx], X_tr[val_idx]
            y_train, y_val = y_tr[train_idx], y_tr[val_idx]

            model = lgb.LGBMRegressor(**param_grid, random_state=random_state)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(period=0)
                ]
            )


            preds = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params