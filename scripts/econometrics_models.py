import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR

def train_var_model(df, var_cols, maxlags=12, ic='aic'):
    model_data = df[var_cols].dropna()
    model = VAR(model_data)
    fitted_model = model.fit(maxlags=maxlags, ic=ic)
    return fitted_model

def arima_model(series, order):
    model = ARIMA(series, order=order)
    result = model.fit()
    return result

def sarimax_model(endog, exog, order, seasonal_order):
    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)
    return result

def grid_search_sarimax(
    series,
    exog=None,
    p_range=None, d_range=None, q_range=None,
    P_range=None, D_range=None, Q_range=None,
    seasonal_period=12,
    verbose=True
):
    """
    Grid search dla modelu SARIMAX z opcjonalnymi zmiennymi egzogenicznymi.

    Args:
        series (pd.Series): endogeniczna zmienna czasowa.
        exog (pd.DataFrame, optional): egzogeniczne zmienne towarzyszące. Default None.
        p_range, d_range, q_range: zakresy parametrów ARIMA.
        P_range, D_range, Q_range: zakresy parametrów sezonowych.
        seasonal_period (int): długość sezonu (np. 12 miesięcy).
        verbose (bool): czy wyświetlać postęp.

    Returns:
        dict: najlepszy model oraz parametry i kryteria AIC, BIC.
    """
    pdq = list(itertools.product(p_range or [0], d_range or [0], q_range or [0]))
    seasonal_pdq = list(itertools.product(P_range or [0], D_range or [0], Q_range or [0]))

    best_aic = np.inf
    best_bic = np.inf
    best_model = None
    best_params = None

    total = len(pdq) * len(seasonal_pdq)
    tested = 0

    if exog is not None:
        exog = exog.loc[series.index]

    for order in pdq:
        for seasonal_order in seasonal_pdq:
            tested += 1
            try:
                model = SARIMAX(
                    endog=series,
                    exog=exog if exog is not None else None,
                    order=order,
                    seasonal_order=seasonal_order + (seasonal_period,),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)

                aic = results.aic
                bic = results.bic

                if verbose:
                    print(f"Tested SARIMAX{order}x{seasonal_order + (seasonal_period,)} - AIC: {aic:.2f}, BIC: {bic:.2f} ({tested}/{total})")

                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_model = results
                    best_params = {
                        "order": order,
                        "seasonal_order": seasonal_order + (seasonal_period,)
                    }
            except Exception as e:
                if verbose:
                    print(f"Failed SARIMAX{order}x{seasonal_order + (seasonal_period,)}: {e}")
                continue

    return {
        "best_model": best_model,
        "order": best_params["order"],
        "seasonal_order": best_params["seasonal_order"],
        "aic": best_aic,
        "bic": best_bic
    }
