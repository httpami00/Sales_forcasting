"""
Model evaluation module for India Car Sales Forecasting.
Calculates MAE, RMSE, WAPE, and MAPE (with safe zero-division handling).
Provides chronological train/test splitting, model comparison tables,
rankings, residual diagnostics, and expanding-window validation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def split_time_series_chronologically(
    df: pd.DataFrame,
    test_size: int = 12
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Split time-series dataframe strictly chronologically without shuffling.
    Returns (train_df, test_df, warning_or_limitation_note).
    """
    clean_df = df.dropna(subset=["Sales"]).sort_values("Date").reset_index(drop=True)
    n = len(clean_df)
    
    if n < 12:
        raise ValueError(f"Dataset has only {n} observations. At least 12 observations required for evaluation.")
        
    warning_note = None
    if test_size >= n:
        test_size = max(1, n // 4)
        warning_note = f"Requested test size exceeded dataset length. Adjusted test size to {test_size} months."
    elif (n - test_size) < 12:
        warning_note = (
            f"Training set has only {n - test_size} months. Seasonal patterns and SARIMA "
            "may have limited statistical stability with less than 24 months of training data."
        )
        
    train_df = clean_df.iloc[:-test_size].copy()
    test_df = clean_df.iloc[-test_size:].copy()
    
    return train_df, test_df, warning_note


def calculate_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Calculate MAE, RMSE, WAPE, and safe MAPE between true and predicted arrays.
    
    WAPE: sum(|y - y_hat|) / sum(y) * 100%
    MAPE: safely handles y_true == 0 by filtering out zero observations and reporting count.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true has {len(y_t)} items, y_pred has {len(y_p)} items.")
    if len(y_t) == 0:
        return {"Model": model_name, "MAE": np.nan, "RMSE": np.nan, "WAPE_Pct": np.nan, "MAPE_Pct": np.nan}
        
    errors = y_t - y_p
    abs_errors = np.abs(errors)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # WAPE (Weighted Absolute Percentage Error)
    sum_true = float(np.sum(y_t))
    if sum_true > 0:
        wape = float((np.sum(abs_errors) / sum_true) * 100.0)
    else:
        wape = np.nan
        
    # Safe MAPE: filter out zero-actual instances to prevent division by zero
    non_zero_mask = (y_t != 0)
    zero_count = int(np.sum(~non_zero_mask))
    
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(errors[non_zero_mask] / y_t[non_zero_mask])) * 100.0)
    else:
        mape = np.nan
        
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE_Pct": round(wape, 2) if not np.isnan(wape) else None,
        "MAPE_Pct": round(mape, 2) if not np.isnan(mape) else None,
        "Sample_Size": len(y_t),
        "Zero_Actuals_Filtered_In_MAPE": zero_count
    }


def compare_and_rank_models(
    metrics_list: List[Dict[str, Any]],
    sort_by: str = "MAE"
) -> pd.DataFrame:
    """
    Generate model comparison table and rank models by chosen metric (lowest is best).
    """
    df_metrics = pd.DataFrame(metrics_list)
    if df_metrics.empty:
        return df_metrics
        
    if sort_by not in df_metrics.columns:
        sort_by = "MAE"
        
    # Sort ascending (lower error is better)
    df_ranked = df_metrics.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    df_ranked["Rank"] = df_ranked.index + 1
    
    # Reorder columns with Rank first
    cols = ["Rank", "Model", "MAE", "RMSE", "WAPE_Pct", "MAPE_Pct", "Sample_Size"]
    existing_cols = [c for c in cols if c in df_ranked.columns]
    remaining = [c for c in df_ranked.columns if c not in existing_cols]
    
    return df_ranked[existing_cols + remaining]


def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> Dict[str, Any]:
    """
    Compute residual statistics and time series for diagnostics:
    residuals = y_true - y_pred (positive means model underpredicted).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p
    
    res_df = pd.DataFrame({
        "Actual": y_t,
        "Predicted": y_p,
        "Residual": residuals,
        "Abs_Error": np.abs(residuals),
        "Pct_Error": np.where(y_t != 0, ((y_t - y_p) / y_t) * 100.0, np.nan)
    })
    if dates is not None:
        res_df["Date"] = pd.to_datetime(dates)
        
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    
    # Durbin-Watson statistic for residual autocorrelation: sum((e_t - e_{t-1})^2) / sum(e_t^2)
    if len(residuals) > 2 and np.sum(residuals ** 2) > 0:
        diff_res = np.diff(residuals)
        dw_stat = float(np.sum(diff_res ** 2) / np.sum(residuals ** 2))
    else:
        dw_stat = np.nan
        
    return {
        "residual_df": res_df,
        "mean_residual": round(mean_res, 2),
        "std_residual": round(std_res, 2),
        "durbin_watson": round(dw_stat, 3) if not np.isnan(dw_stat) else None,
        "underpredicted_count": int(np.sum(residuals > 0)),
        "overpredicted_count": int(np.sum(residuals < 0)),
        "exact_count": int(np.sum(residuals == 0))
    }


def perform_expanding_window_cv(
    df: pd.DataFrame,
    initial_train_size: int = 36,
    step_size: int = 6,
    horizon: int = 6,
    models_to_test: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform expanding-window time-series cross-validation.
    Simulates rolling production deployments across historical time folds.
    """
    from .forecasting import forecast_naive, forecast_seasonal_naive, forecast_arima, forecast_sarima
    
    clean_df = df.dropna(subset=["Sales"]).sort_values("Date").reset_index(drop=True)
    total_len = len(clean_df)
    
    if total_len < initial_train_size + horizon:
        return pd.DataFrame()
        
    if models_to_test is None:
        models_to_test = ["Naive", "Seasonal Naive", "ARIMA(1,1,1)", "SARIMA"]
        
    fold_records = []
    fold_idx = 1
    current_split = initial_train_size
    
    while current_split + horizon <= total_len:
        train_slice = clean_df.iloc[:current_split].copy()
        test_slice = clean_df.iloc[current_split:current_split + horizon].copy()
        
        train_series = train_slice.set_index("Date")["Sales"]
        test_actuals = test_slice["Sales"].values
        test_dates = test_slice["Date"].values
        
        # Test Naive
        if "Naive" in models_to_test:
            try:
                res = forecast_naive(train_series, steps=horizon, future_dates=test_dates)
                m = calculate_evaluation_metrics(test_actuals, res.point_forecast, "Naive")
                m["Fold"] = fold_idx
                m["Train_End"] = train_slice["Date"].max().strftime("%Y-%m")
                m["Test_Period"] = f"{test_slice['Date'].min().strftime('%Y-%m')} to {test_slice['Date'].max().strftime('%Y-%m')}"
                fold_records.append(m)
            except Exception:
                pass
                
        # Test Seasonal Naive
        if "Seasonal Naive" in models_to_test and len(train_series) >= 12:
            try:
                res = forecast_seasonal_naive(train_series, steps=horizon, future_dates=test_dates)
                m = calculate_evaluation_metrics(test_actuals, res.point_forecast, "Seasonal Naive")
                m["Fold"] = fold_idx
                m["Train_End"] = train_slice["Date"].max().strftime("%Y-%m")
                m["Test_Period"] = f"{test_slice['Date'].min().strftime('%Y-%m')} to {test_slice['Date'].max().strftime('%Y-%m')}"
                fold_records.append(m)
            except Exception:
                pass
                
        # Test ARIMA
        if "ARIMA(1,1,1)" in models_to_test and len(train_series) >= 12:
            try:
                res = forecast_arima(train_series, steps=horizon, order=(1, 1, 1), future_dates=test_dates)
                m = calculate_evaluation_metrics(test_actuals, res.point_forecast, "ARIMA(1,1,1)")
                m["Fold"] = fold_idx
                m["Train_End"] = train_slice["Date"].max().strftime("%Y-%m")
                m["Test_Period"] = f"{test_slice['Date'].min().strftime('%Y-%m')} to {test_slice['Date'].max().strftime('%Y-%m')}"
                fold_records.append(m)
            except Exception:
                pass
                
        # Test SARIMA
        if "SARIMA" in models_to_test and len(train_series) >= 24:
            try:
                res = forecast_sarima(train_series, steps=horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), future_dates=test_dates)
                m = calculate_evaluation_metrics(test_actuals, res.point_forecast, "SARIMA")
                m["Fold"] = fold_idx
                m["Train_End"] = train_slice["Date"].max().strftime("%Y-%m")
                m["Test_Period"] = f"{test_slice['Date'].min().strftime('%Y-%m')} to {test_slice['Date'].max().strftime('%Y-%m')}"
                fold_records.append(m)
            except Exception:
                pass

        fold_idx += 1
        current_split += step_size
        
    return pd.DataFrame(fold_records)
