"""
Unit tests for forecasting models: Naive, Seasonal Naive, ARIMA, SARIMA, Prophet.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    forecast_naive,
    forecast_seasonal_naive,
    forecast_arima,
    forecast_sarima,
    forecast_prophet,
    generate_future_forecast_dates,
    run_all_available_models,
    PROPHET_AVAILABLE
)


@pytest.fixture
def sample_monthly_sales_series():
    # 36 months of realistic monthly automotive dispatches
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")
    # Base level + upward trend + seasonal pattern
    base = 250000
    trend = np.linspace(0, 50000, 36)
    seasonality = np.tile([0, -5000, 10000, -10000, -8000, -15000, 5000, 12000, 20000, 35000, 15000, -10000], 3)
    sales = base + trend + seasonality
    return pd.Series(sales, index=dates)


def test_generate_future_forecast_dates():
    last_date = pd.Timestamp("2024-12-01")
    future_dates = generate_future_forecast_dates(last_date, steps=6)
    assert len(future_dates) == 6
    assert future_dates[0] == pd.Timestamp("2025-01-01")
    assert future_dates[5] == pd.Timestamp("2025-06-01")


def test_naive_forecast(sample_monthly_sales_series):
    steps = 6
    last_val = sample_monthly_sales_series.iloc[-1]
    res = forecast_naive(sample_monthly_sales_series, steps=steps)
    
    assert len(res.point_forecast) == steps
    assert (res.point_forecast == last_val).all()
    assert res.lower_bound is not None
    assert res.upper_bound is not None
    
    # Test conversion to dataframe
    df_fc = res.to_dataframe(last_observed_value=last_val)
    assert len(df_fc) == steps
    assert "Forecast" in df_fc.columns
    assert "MoM_Change_Pct" in df_fc.columns


def test_seasonal_naive_forecast(sample_monthly_sales_series):
    steps = 12
    res = forecast_seasonal_naive(sample_monthly_sales_series, steps=steps, seasonal_period=12)
    assert len(res.point_forecast) == steps
    # Point forecast should match the last 12 values of the historical series
    expected = sample_monthly_sales_series.iloc[-12:].values
    np.testing.assert_allclose(res.point_forecast, expected, rtol=1e-5)


def test_arima_forecast(sample_monthly_sales_series):
    steps = 6
    res = forecast_arima(sample_monthly_sales_series, steps=steps, order=(1, 1, 1))
    assert len(res.point_forecast) == steps
    assert (res.point_forecast > 0).all()
    assert len(res.lower_bound) == steps
    assert len(res.upper_bound) == steps
    assert "aic" in res.model_params


def test_sarima_forecast(sample_monthly_sales_series):
    steps = 6
    res = forecast_sarima(
        sample_monthly_sales_series,
        steps=steps,
        order=(1, 0, 0),
        seasonal_order=(0, 1, 0, 12)
    )
    assert len(res.point_forecast) == steps
    assert (res.point_forecast > 0).all()
    assert len(res.lower_bound) == steps
    assert len(res.upper_bound) == steps


def test_prophet_forecast_if_installed(sample_monthly_sales_series):
    if not PROPHET_AVAILABLE:
        pytest.skip("Prophet is not installed in the environment.")
        
    steps = 6
    res = forecast_prophet(sample_monthly_sales_series, steps=steps)
    assert len(res.point_forecast) == steps
    assert (res.point_forecast > 0).all()
    assert res.lower_bound is not None
    assert res.upper_bound is not None


def test_run_all_available_models(sample_monthly_sales_series):
    all_res = run_all_available_models(sample_monthly_sales_series, steps=3)
    assert "naive" in all_res
    assert "seasonal_naive" in all_res
    assert "arima" in all_res
    assert "sarima" in all_res
    
    # Check that at least Naive, Seasonal Naive, ARIMA, and SARIMA succeeded
    for key in ["naive", "seasonal_naive", "arima", "sarima"]:
        res_obj, err = all_res[key]
        assert err is None, f"Model {key} failed with error: {err}"
        assert res_obj is not None
