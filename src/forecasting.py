"""
Time-series forecasting module for India Car Sales.
Implements:
1. Model A: Naive Forecast (last observation carried forward)
2. Model B: Seasonal Naive Forecast (same month previous year, t - 12)
3. Model C: ARIMA (Statsmodels autoregressive integrated moving average)
4. Model D: SARIMA (Seasonal ARIMA with period s=12, optional exogenous variables)
5. Model E: Prophet (Bayesian additive time-series model with yearly seasonality)
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Check Prophet availability
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    PROPHET_ERROR_MSG = None
except Exception as e:
    PROPHET_AVAILABLE = False
    PROPHET_ERROR_MSG = str(e)


class ForecastResult:
    """Standardized container for forecast outputs."""
    def __init__(
        self,
        model_name: str,
        forecast_dates: pd.DatetimeIndex,
        point_forecast: np.ndarray,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        model_params: Optional[Dict[str, Any]] = None,
        training_history_length: int = 0
    ):
        self.model_name = model_name
        self.forecast_dates = pd.to_datetime(forecast_dates)
        self.point_forecast = np.asarray(point_forecast, dtype=float)
        # Ensure non-negative car sales predictions
        self.point_forecast = np.clip(self.point_forecast, 0, None)
        
        if lower_bound is not None:
            self.lower_bound = np.clip(np.asarray(lower_bound, dtype=float), 0, None)
        else:
            self.lower_bound = None
            
        if upper_bound is not None:
            self.upper_bound = np.clip(np.asarray(upper_bound, dtype=float), 0, None)
        else:
            self.upper_bound = None
            
        self.model_params = model_params or {}
        self.training_history_length = training_history_length

    def to_dataframe(
        self,
        last_observed_date: Optional[pd.Timestamp] = None,
        last_observed_value: Optional[float] = None,
        historical_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Convert forecast into display/export DataFrame with MoM and YoY changes.
        """
        df_fc = pd.DataFrame({
            "Date": self.forecast_dates,
            "Forecast": np.round(self.point_forecast).astype(int),
            "Lower_Bound": np.round(self.lower_bound).astype(int) if self.lower_bound is not None else None,
            "Upper_Bound": np.round(self.upper_bound).astype(int) if self.upper_bound is not None else None,
            "Model": self.model_name
        })
        
        # Calculate MoM change
        # First forecast month compares to last historical observation
        mom_changes = []
        for i in range(len(df_fc)):
            if i == 0:
                if last_observed_value is not None and last_observed_value > 0:
                    pct = ((df_fc.iloc[0]["Forecast"] - last_observed_value) / last_observed_value) * 100.0
                else:
                    pct = 0.0
            else:
                prev_fc = df_fc.iloc[i - 1]["Forecast"]
                pct = ((df_fc.iloc[i]["Forecast"] - prev_fc) / prev_fc) * 100.0 if prev_fc > 0 else 0.0
            mom_changes.append(pct)
        df_fc["MoM_Change_Pct"] = np.round(mom_changes, 2)
        
        # Calculate YoY change (compare forecast date with same month 1 year prior)
        yoy_changes = []
        for _, row in df_fc.iterrows():
            target_past = row["Date"] - pd.DateOffset(years=1)
            past_val = None
            if historical_df is not None and "Date" in historical_df.columns and "Sales" in historical_df.columns:
                match = historical_df[historical_df["Date"] == target_past]
                if not match.empty and pd.notnull(match.iloc[0]["Sales"]):
                    past_val = float(match.iloc[0]["Sales"])
            
            if past_val is not None and past_val > 0:
                pct_yoy = ((row["Forecast"] - past_val) / past_val) * 100.0
                yoy_changes.append(np.round(pct_yoy, 2))
            else:
                yoy_changes.append(None)
                
        df_fc["YoY_Change_Pct"] = yoy_changes
        return df_fc


def generate_future_forecast_dates(
    last_date: pd.Timestamp,
    steps: int = 12
) -> pd.DatetimeIndex:
    """
    Generate future monthly sequence ('MS') starting immediately after last_date.
    """
    start_future = last_date + pd.DateOffset(months=1)
    # Ensure start is on day 1
    start_future = start_future.replace(day=1)
    return pd.date_range(start=start_future, periods=steps, freq="MS")


# ==============================================================================
# Model Implementations
# ==============================================================================

def forecast_naive(
    train_series: pd.Series,
    steps: int = 12,
    future_dates: Optional[pd.DatetimeIndex] = None
) -> ForecastResult:
    """
    Model A: Naive Forecast (Random Walk / Flat persistence).
    Forecast is the last observed value: y_hat_{t+h} = y_t.
    """
    clean_series = train_series.dropna()
    if len(clean_series) == 0:
        raise ValueError("Cannot fit Naive model on empty training data.")
        
    last_val = float(clean_series.iloc[-1])
    preds = np.full(steps, last_val)
    
    # Simple naive standard error from historical residuals
    if len(clean_series) > 1:
        diffs = clean_series.diff().dropna()
        residual_std = float(diffs.std())
        # Prediction interval grows with sqrt(h)
        h_array = np.arange(1, steps + 1)
        half_width = 1.96 * residual_std * np.sqrt(h_array)
        lower = preds - half_width
        upper = preds + half_width
    else:
        lower = None
        upper = None
        
    if future_dates is None:
        last_date = clean_series.index[-1] if isinstance(clean_series.index, pd.DatetimeIndex) else pd.Timestamp("2024-12-01")
        future_dates = generate_future_forecast_dates(last_date, steps)
        
    return ForecastResult(
        model_name="Naive (Last Value)",
        forecast_dates=future_dates,
        point_forecast=preds,
        lower_bound=lower,
        upper_bound=upper,
        model_params={"strategy": "last_observation", "last_value": last_val},
        training_history_length=len(clean_series)
    )


def forecast_seasonal_naive(
    train_series: pd.Series,
    steps: int = 12,
    seasonal_period: int = 12,
    future_dates: Optional[pd.DatetimeIndex] = None
) -> ForecastResult:
    """
    Model B: Seasonal Naive Forecast.
    Uses observation from the same seasonal period of the previous year (t + h - 12).
    """
    clean_series = train_series.dropna()
    n = len(clean_series)
    if n < seasonal_period:
        # Fallback to naive if not enough history for 1 full cycle
        warnings.warn(f"Seasonal Naive requires at least {seasonal_period} observations. Falling back to Naive.")
        res = forecast_naive(train_series, steps, future_dates)
        res.model_name = "Seasonal Naive (Fallback to Naive)"
        return res
        
    # Get last `seasonal_period` observations
    seasonal_slice = clean_series.iloc[-seasonal_period:].values
    
    # Repeat seasonal pattern for required steps
    repeats = int(np.ceil(steps / seasonal_period))
    extended = np.tile(seasonal_slice, repeats)[:steps]
    preds = extended.astype(float)
    
    # Seasonal naive standard error from seasonal differences
    if n > seasonal_period:
        seasonal_diffs = (clean_series - clean_series.shift(seasonal_period)).dropna()
        s_std = float(seasonal_diffs.std()) if not seasonal_diffs.empty else float(clean_series.std())
        half_width = 1.96 * s_std
        lower = preds - half_width
        upper = preds + half_width
    else:
        lower = None
        upper = None
        
    if future_dates is None:
        last_date = clean_series.index[-1] if isinstance(clean_series.index, pd.DatetimeIndex) else pd.Timestamp("2024-12-01")
        future_dates = generate_future_forecast_dates(last_date, steps)
        
    return ForecastResult(
        model_name="Seasonal Naive (t-12)",
        forecast_dates=future_dates,
        point_forecast=preds,
        lower_bound=lower,
        upper_bound=upper,
        model_params={"seasonal_period": seasonal_period},
        training_history_length=n
    )


def forecast_arima(
    train_series: pd.Series,
    steps: int = 12,
    order: Tuple[int, int, int] = (1, 1, 1),
    future_dates: Optional[pd.DatetimeIndex] = None
) -> ForecastResult:
    """
    Model C: ARIMA (p, d, q) via Statsmodels.
    """
    clean_series = train_series.dropna()
    if len(clean_series) < 10:
        raise ValueError("ARIMA requires at least 10 observations to estimate parameters reliably.")
        
    # Fit ARIMA model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(clean_series, order=order)
        fitted = model.fit()
        forecast_res = fitted.get_forecast(steps=steps)
        preds = forecast_res.predicted_mean.values
        conf_int = forecast_res.conf_int(alpha=0.05)
        lower = conf_int.iloc[:, 0].values if hasattr(conf_int, "iloc") else conf_int[:, 0]
        upper = conf_int.iloc[:, 1].values if hasattr(conf_int, "iloc") else conf_int[:, 1]
        
    if future_dates is None:
        last_date = clean_series.index[-1] if isinstance(clean_series.index, pd.DatetimeIndex) else pd.Timestamp("2024-12-01")
        future_dates = generate_future_forecast_dates(last_date, steps)
        
    return ForecastResult(
        model_name=f"ARIMA{order}",
        forecast_dates=future_dates,
        point_forecast=preds,
        lower_bound=lower,
        upper_bound=upper,
        model_params={"order": order, "aic": float(fitted.aic), "bic": float(fitted.bic)},
        training_history_length=len(clean_series)
    )


def forecast_sarima(
    train_series: pd.Series,
    steps: int = 12,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    exog_train: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    future_dates: Optional[pd.DatetimeIndex] = None
) -> ForecastResult:
    """
    Model D: Seasonal ARIMA (p, d, q) x (P, D, Q, s) via Statsmodels.
    Optionally supports exogenous regressors (SARIMAX).
    """
    clean_series = train_series.dropna()
    s = seasonal_order[3]
    if len(clean_series) < 2 * s:
        # If dataset is too short for seasonal difference, fall back gracefully
        warnings.warn(f"Dataset has {len(clean_series)} rows; SARIMA recommended >= {2 * s}. Adjusting seasonal order.")
        seasonal_order = (0, 0, 0, 0)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                clean_series,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False, maxiter=200)
            forecast_res = fitted.get_forecast(steps=steps, exog=exog_future)
            preds = forecast_res.predicted_mean.values
            conf_int = forecast_res.conf_int(alpha=0.05)
            lower = conf_int.iloc[:, 0].values if hasattr(conf_int, "iloc") else conf_int[:, 0]
            upper = conf_int.iloc[:, 1].values if hasattr(conf_int, "iloc") else conf_int[:, 1]
            aic_val = float(fitted.aic)
            bic_val = float(fitted.bic)
        except Exception as e:
            # Fallback to simpler SARIMA specification if convergence failed
            model = SARIMAX(clean_series, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12))
            fitted = model.fit(disp=False, maxiter=100)
            forecast_res = fitted.get_forecast(steps=steps)
            preds = forecast_res.predicted_mean.values
            conf_int = forecast_res.conf_int(alpha=0.05)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
            aic_val = float(fitted.aic)
            bic_val = float(fitted.bic)
            
    if future_dates is None:
        last_date = clean_series.index[-1] if isinstance(clean_series.index, pd.DatetimeIndex) else pd.Timestamp("2024-12-01")
        future_dates = generate_future_forecast_dates(last_date, steps)
        
    name = f"SARIMA{order}x{seasonal_order}"
    if exog_train is not None:
        name += " + Exog"
        
    return ForecastResult(
        model_name=name,
        forecast_dates=future_dates,
        point_forecast=preds,
        lower_bound=lower,
        upper_bound=upper,
        model_params={"order": order, "seasonal_order": seasonal_order, "aic": aic_val, "bic": bic_val},
        training_history_length=len(clean_series)
    )


def forecast_prophet(
    train_series: pd.Series,
    steps: int = 12,
    future_dates: Optional[pd.DatetimeIndex] = None
) -> ForecastResult:
    """
    Model E: Prophet forecasting with yearly seasonality.
    Requires prophet package. Handled gracefully if unavailable.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError(f"Prophet is not available in the current environment: {PROPHET_ERROR_MSG}")
        
    clean_series = train_series.dropna()
    if len(clean_series) < 12:
        raise ValueError("Prophet requires at least 12 months of observations to estimate yearly seasonality.")
        
    # Prepare Prophet format dataframe: 'ds', 'y'
    dates = clean_series.index if isinstance(clean_series.index, pd.DatetimeIndex) else pd.date_range("2018-01-01", periods=len(clean_series), freq="MS")
    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(dates),
        "y": clean_series.values
    })
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive"
        )
        m.fit(df_prophet)
        
        future_df = m.make_future_dataframe(periods=steps, freq="MS", include_history=False)
        forecast = m.predict(future_df)
        
    preds = forecast["yhat"].values
    lower = forecast["yhat_lower"].values
    upper = forecast["yhat_upper"].values
    
    if future_dates is None:
        future_dates = pd.to_datetime(forecast["ds"])
        
    return ForecastResult(
        model_name="Prophet (Additive)",
        forecast_dates=future_dates,
        point_forecast=preds,
        lower_bound=lower,
        upper_bound=upper,
        model_params={"yearly_seasonality": True, "mode": "additive"},
        training_history_length=len(clean_series)
    )


def run_all_available_models(
    train_series: pd.Series,
    steps: int = 12,
    future_dates: Optional[pd.DatetimeIndex] = None,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_seasonal: Tuple[int, int, int, int] = (1, 1, 1, 12),
    exog_train: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None
) -> Dict[str, Tuple[Optional[ForecastResult], Optional[str]]]:
    """
    Run all forecasting models across the specified dataset.
    Returns dictionary mapping model key -> (ForecastResult, error_message_if_any).
    """
    results: Dict[str, Tuple[Optional[ForecastResult], Optional[str]]] = {}
    
    # 1. Naive
    try:
        results["naive"] = (forecast_naive(train_series, steps=steps, future_dates=future_dates), None)
    except Exception as e:
        results["naive"] = (None, str(e))
        
    # 2. Seasonal Naive
    try:
        results["seasonal_naive"] = (forecast_seasonal_naive(train_series, steps=steps, future_dates=future_dates), None)
    except Exception as e:
        results["seasonal_naive"] = (None, str(e))
        
    # 3. ARIMA
    try:
        results["arima"] = (forecast_arima(train_series, steps=steps, order=arima_order, future_dates=future_dates), None)
    except Exception as e:
        results["arima"] = (None, str(e))
        
    # 4. SARIMA
    try:
        results["sarima"] = (
            forecast_sarima(
                train_series,
                steps=steps,
                order=sarima_order,
                seasonal_order=sarima_seasonal,
                exog_train=exog_train,
                exog_future=exog_future,
                future_dates=future_dates
            ),
            None
        )
    except Exception as e:
        results["sarima"] = (None, str(e))
        
    # 5. Prophet
    if PROPHET_AVAILABLE:
        try:
            results["prophet"] = (forecast_prophet(train_series, steps=steps, future_dates=future_dates), None)
        except Exception as e:
            results["prophet"] = (None, str(e))
    else:
        results["prophet"] = (None, f"Prophet not installed: {PROPHET_ERROR_MSG}")
        
    return results
