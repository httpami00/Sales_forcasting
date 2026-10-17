"""
Exploratory Data Analysis (EDA) module for India Car Sales.
Calculates sales KPIs, time-series metrics, rolling windows, YoY/MoM growth,
calendar seasonality, seasonal decomposition, and statistical anomaly detection.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def calculate_sales_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate executive sales KPIs across the dataset.
    """
    valid_df = df.dropna(subset=["Sales"]).sort_values("Date").reset_index(drop=True)
    if len(valid_df) == 0:
        return {}
    
    total_sales = float(valid_df["Sales"].sum())
    avg_sales = float(valid_df["Sales"].mean())
    std_sales = float(valid_df["Sales"].std())
    median_sales = float(valid_df["Sales"].median())
    
    max_row = valid_df.loc[valid_df["Sales"].idxmax()]
    min_row = valid_df.loc[valid_df["Sales"].idxmin()]
    latest_row = valid_df.iloc[-1]
    
    # Calculate MoM and YoY for the latest observation
    latest_sales = float(latest_row["Sales"])
    latest_date = latest_row["Date"]
    
    mom_change_pct = None
    yoy_change_pct = None
    
    if len(valid_df) >= 2:
        prev_month_sales = float(valid_df.iloc[-2]["Sales"])
        if prev_month_sales > 0:
            mom_change_pct = ((latest_sales - prev_month_sales) / prev_month_sales) * 100.0
            
    # YoY: look for observation exactly 12 months prior
    target_yoy_date = latest_date - pd.DateOffset(years=1)
    yoy_match = valid_df[valid_df["Date"] == target_yoy_date]
    if not yoy_match.empty:
        prev_year_sales = float(yoy_match.iloc[0]["Sales"])
        if prev_year_sales > 0:
            yoy_change_pct = ((latest_sales - prev_year_sales) / prev_year_sales) * 100.0

    return {
        "total_historical_sales": total_sales,
        "average_monthly_sales": avg_sales,
        "median_monthly_sales": median_sales,
        "std_monthly_sales": std_sales,
        "highest_sales_month": max_row["Date"].strftime("%B %Y"),
        "highest_sales_value": float(max_row["Sales"]),
        "lowest_sales_month": min_row["Date"].strftime("%B %Y"),
        "lowest_sales_value": float(min_row["Sales"]),
        "latest_recorded_month": latest_row["Date"].strftime("%B %Y"),
        "latest_recorded_sales": latest_sales,
        "latest_mom_change_pct": mom_change_pct,
        "latest_yoy_change_pct": yoy_change_pct,
        "total_months_count": len(valid_df),
        "start_month": valid_df["Date"].min().strftime("%B %Y"),
        "end_month": valid_df["Date"].max().strftime("%B %Y")
    }


def compute_growth_and_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages (3, 6, 12 months), MoM % change, and YoY % change.
    """
    ts = df.copy().sort_values("Date").reset_index(drop=True)
    
    # Rolling averages
    ts["Rolling_3M"] = ts["Sales"].rolling(window=3, min_periods=1).mean()
    ts["Rolling_6M"] = ts["Sales"].rolling(window=6, min_periods=1).mean()
    ts["Rolling_12M"] = ts["Sales"].rolling(window=12, min_periods=1).mean()
    
    # MoM percentage growth
    ts["MoM_Growth_Pct"] = ts["Sales"].pct_change(periods=1) * 100.0
    
    # YoY percentage growth (12 months prior)
    ts["YoY_Growth_Pct"] = ts["Sales"].pct_change(periods=12) * 100.0
    
    return ts


def get_recent_slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Extract full, recent 5-year, and recent 2-year time-series slices.
    """
    ts = df.copy().sort_values("Date").reset_index(drop=True)
    if len(ts) == 0:
        return {"full": ts, "5_year": ts, "2_year": ts}
    
    max_date = ts["Date"].max()
    date_5y = max_date - pd.DateOffset(years=5)
    date_2y = max_date - pd.DateOffset(years=2)
    
    return {
        "full": ts,
        "5_year": ts[ts["Date"] >= date_5y].copy(),
        "2_year": ts[ts["Date"] >= date_2y].copy()
    }


def analyze_monthly_seasonality(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze seasonality patterns:
    1. Summary statistics grouped by calendar month (1 to 12).
    2. Heatmap matrix (Years vs Months).
    """
    ts = df.dropna(subset=["Sales"]).copy()
    ts["Year"] = ts["Date"].dt.year
    ts["Month_Num"] = ts["Date"].dt.month
    ts["Month_Name"] = ts["Date"].dt.strftime("%b")
    
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Monthly aggregates
    monthly_agg = ts.groupby(["Month_Num", "Month_Name"], as_index=False)["Sales"].agg(
        Mean="mean",
        Median="median",
        Min="min",
        Max="max",
        Std="std",
        Count="count"
    ).sort_values("Month_Num")
    
    # Pivot for Heatmap (Year rows, Month columns)
    pivot_heatmap = ts.pivot_table(index="Year", columns="Month_Name", values="Sales", aggfunc="mean")
    # Reindex columns to natural calendar order
    available_cols = [m for m in month_order if m in pivot_heatmap.columns]
    pivot_heatmap = pivot_heatmap[available_cols]
    
    return monthly_agg, pivot_heatmap


def perform_seasonal_decomposition(
    df: pd.DataFrame,
    model: str = "additive"
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Perform classical seasonal decomposition into Trend, Seasonal, and Residual.
    Requires at least 24 complete observations (2 full annual cycles).
    """
    clean_series = df.dropna(subset=["Sales"]).sort_values("Date").copy()
    
    if len(clean_series) < 24:
        return None, (
            f"Seasonal decomposition requires at least 24 consecutive monthly observations "
            f"(2 full 12-month annual cycles). Current dataset has {len(clean_series)} valid rows."
        )
    
    # If multiplicative and series contains values <= 0 (e.g. April 2020), switch to additive
    if model == "multiplicative" and (clean_series["Sales"] <= 0).any():
        model = "additive"
        
    indexed_series = clean_series.set_index("Date")["Sales"].asfreq("MS")
    # If any NaNs remain in indexed series, linearly interpolate for decomposition calculation
    if indexed_series.isna().any():
        indexed_series = indexed_series.interpolate(method="time")
        
    try:
        decomp = seasonal_decompose(indexed_series, model=model, period=12)
        return decomp, None
    except Exception as e:
        return None, f"Decomposition failed: {str(e)}"


def detect_statistical_anomalies(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identify statistical anomalies in monthly car sales.
    Methods:
    - 'iqr': Outliers beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR of residuals or raw sales.
    - 'zscore': Observations where absolute Z-score > threshold (default 2.5).
    
    Returns DataFrame with columns:
    ['Date', 'Sales', 'Is_Anomaly', 'Anomaly_Score', 'Anomaly_Direction', 'Anomaly_Reason']
    Note: Does NOT delete anomalies.
    """
    ts = df.copy().sort_values("Date").reset_index(drop=True)
    sales = ts["Sales"].copy()
    
    ts["Is_Anomaly"] = False
    ts["Anomaly_Score"] = 0.0
    ts["Anomaly_Direction"] = "Normal"
    ts["Anomaly_Reason"] = ""
    
    if len(ts.dropna(subset=["Sales"])) < 5:
        return ts
    
    if method == "zscore":
        mean_val = sales.mean()
        std_val = sales.std()
        if std_val > 0:
            z_scores = (sales - mean_val) / std_val
            ts["Anomaly_Score"] = z_scores.abs()
            
            high_mask = z_scores > threshold
            low_mask = z_scores < -threshold
            
            ts.loc[high_mask, "Is_Anomaly"] = True
            ts.loc[high_mask, "Anomaly_Direction"] = "High"
            ts.loc[high_mask, "Anomaly_Reason"] = ts.loc[high_mask].apply(
                lambda r: f"Z-score {z_scores[r.name]:.2f} > +{threshold} (Unusually high volume)", axis=1
            )
            
            ts.loc[low_mask, "Is_Anomaly"] = True
            ts.loc[low_mask, "Anomaly_Direction"] = "Low"
            ts.loc[low_mask, "Anomaly_Reason"] = ts.loc[low_mask].apply(
                lambda r: f"Z-score {z_scores[r.name]:.2f} < -{threshold} (Unusually depressed volume / shock)", axis=1
            )
            
    else:  # IQR method (default)
        # Using rolling median residual IQR to respect underlying trend
        rolling_median = sales.rolling(window=12, min_periods=3, center=True).median().fillna(sales.median())
        residuals = sales - rolling_median
        
        q25 = residuals.quantile(0.25)
        q75 = residuals.quantile(0.75)
        iqr = q75 - q25
        
        lower_bound = q25 - (threshold * iqr)
        upper_bound = q75 + (threshold * iqr)
        
        low_mask = residuals < lower_bound
        high_mask = residuals > upper_bound
        
        ts["Anomaly_Score"] = (residuals - residuals.median()).abs() / (iqr if iqr > 0 else 1.0)
        
        ts.loc[high_mask, "Is_Anomaly"] = True
        ts.loc[high_mask, "Anomaly_Direction"] = "High"
        ts.loc[high_mask, "Anomaly_Reason"] = "Deviation significantly exceeds upper IQR fence (festive peak / demand spike)"
        
        ts.loc[low_mask, "Is_Anomaly"] = True
        ts.loc[low_mask, "Anomaly_Direction"] = "Low"
        ts.loc[low_mask, "Anomaly_Reason"] = "Deviation significantly below lower IQR fence (lockdown, supply crisis, or severe shock)"
        
    return ts
