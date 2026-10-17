"""
Unit tests for model evaluation and metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_evaluation_metrics,
    compare_and_rank_models,
    compute_residual_diagnostics,
    split_time_series_chronologically
)


def test_chronological_split():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"Date": dates, "Sales": np.arange(100, 124)})
    
    train, test, warn = split_time_series_chronologically(df, test_size=6)
    assert len(train) == 18
    assert len(test) == 6
    assert train["Date"].max() < test["Date"].min()
    assert test["Date"].iloc[-1] == dates[-1]


def test_metric_calculations():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    
    m = calculate_evaluation_metrics(y_true, y_pred, "TestModel")
    # MAE = (10 + 10 + 30) / 3 = 16.67
    assert pytest.approx(m["MAE"], 0.01) == 16.67
    # RMSE = sqrt((100 + 100 + 900) / 3) = sqrt(1100 / 3) = 19.15
    assert pytest.approx(m["RMSE"], 0.01) == 19.15
    # WAPE = (10 + 10 + 30) / (100 + 200 + 300) * 100 = 50 / 600 * 100 = 8.33%
    assert pytest.approx(m["WAPE_Pct"], 0.01) == 8.33
    # MAPE = (10/100 + 10/200 + 30/300) / 3 * 100 = (0.10 + 0.05 + 0.10) / 3 * 100 = 8.33%
    assert pytest.approx(m["MAPE_Pct"], 0.01) == 8.33


def test_safe_mape_with_zero_actual():
    # If actual contains 0 (like April 2020 lockdown), MAPE must not blow up with inf/nan
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 190.0])
    
    m = calculate_evaluation_metrics(y_true, y_pred, "ZeroActualModel")
    assert not np.isinf(m["MAPE_Pct"])
    assert not np.isnan(m["MAPE_Pct"])
    assert m["Zero_Actuals_Filtered_In_MAPE"] == 1
    # MAE and WAPE should remain completely valid
    assert pytest.approx(m["MAE"], 0.01) == 10.0
    assert pytest.approx(m["WAPE_Pct"], 0.01) == 10.0


def test_compare_and_rank_models():
    metrics = [
        {"Model": "Model B", "MAE": 25.0, "RMSE": 30.0, "WAPE_Pct": 5.0, "MAPE_Pct": 5.0},
        {"Model": "Model A", "MAE": 15.0, "RMSE": 20.0, "WAPE_Pct": 3.0, "MAPE_Pct": 3.0},
        {"Model": "Model C", "MAE": 35.0, "RMSE": 40.0, "WAPE_Pct": 7.0, "MAPE_Pct": 7.0},
    ]
    ranked = compare_and_rank_models(metrics, sort_by="MAE")
    assert ranked.iloc[0]["Model"] == "Model A"
    assert ranked.iloc[0]["Rank"] == 1
    assert ranked.iloc[1]["Model"] == "Model B"
    assert ranked.iloc[2]["Model"] == "Model C"


def test_residual_diagnostics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([90.0, 210.0, 290.0, 410.0])
    dates = pd.date_range("2023-01-01", periods=4, freq="MS")
    
    diag = compute_residual_diagnostics(y_true, y_pred, dates)
    assert len(diag["residual_df"]) == 4
    assert diag["underpredicted_count"] == 2  # actual > pred
    assert diag["overpredicted_count"] == 2  # actual < pred
    assert diag["durbin_watson"] is not None
