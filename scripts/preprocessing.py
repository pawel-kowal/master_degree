import pandas as pd
from statsmodels.tsa.stattools import adfuller

def load_data(filepath):
    df = pd.read_excel(filepath, parse_dates=['rok_miesiac'])
    df['rok_miesiac'] = pd.to_datetime(df['rok_miesiac'], format='%Y%m')
    df.set_index('rok_miesiac', inplace=True)
    return df

def test_stationarity(series, alpha=0.05):
    result = adfuller(series.dropna())
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Stationary': result[1] <= alpha
    }

def create_lagged_features(df, variables, lags):
    df_lagged = df.copy()
    for var in variables:
        for lag in range(1, lags+1):
            df_lagged[f"{var}_lag{lag}"] = df_lagged[var].shift(lag)
    return df_lagged.dropna()
