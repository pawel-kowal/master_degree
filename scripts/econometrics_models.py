import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests

def ols_regression(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model

def arima_model(series, order):
    model = ARIMA(series, order=order)
    result = model.fit()
    return result

def sarimax_model(endog, exog, order, seasonal_order):
    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)
    return result

def test_heteroskedasticity(residuals, exog):
    return het_breuschpagan(residuals, exog)

def test_autocorrelation(residuals, lags=12):
    return acorr_ljungbox(residuals, lags=lags, return_df=True)

def granger_test(df, variables, max_lag):
    results = {}
    for i in variables:
        for j in variables:
            if i != j:
                test = grangercausalitytests(df[[i, j]].dropna(), maxlag=max_lag, verbose=False)
                p_values = [test[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag+1)]
                results[f"{j} -> {i}"] = p_values
    return results
