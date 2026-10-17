"""
Unit tests for data loader and preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_numeric_sales,
    parse_dates_to_month_start,
    detect_duplicates,
    resolve_duplicates,
    identify_missing_months_sequence,
    preprocess_car_sales_data
)
from src.data_loader import detect_suggested_columns, map_and_standardize_columns


def test_clean_numeric_sales():
    raw_series = pd.Series(["346,080", " ₹285,477 ", "300000 units", 150000, "invalid", None])
    cleaned = clean_numeric_sales(raw_series)
    assert cleaned.iloc[0] == 346080.0
    assert cleaned.iloc[1] == 285477.0
    assert cleaned.iloc[2] == 300000.0
    assert cleaned.iloc[3] == 150000.0
    assert np.isnan(cleaned.iloc[4])
    assert np.isnan(cleaned.iloc[5])


def test_parse_dates_to_month_start():
    raw_dates = pd.Series(["2024-01-15", "2024/02/28", "2024-03-01", "invalid_date"])
    parsed = parse_dates_to_month_start(raw_dates)
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01")
    assert parsed.iloc[1] == pd.Timestamp("2024-02-01")
    assert parsed.iloc[2] == pd.Timestamp("2024-03-01")
    assert pd.isna(parsed.iloc[3])


def test_duplicate_detection_and_resolution():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01"]),
        "Sales": [100.0, 150.0, 200.0]
    })
    count, dup_months = detect_duplicates(df)
    assert count == 2
    assert "2023-01" in dup_months
    
    # Test keep_first
    df_first = resolve_duplicates(df, method="keep_first")
    assert len(df_first) == 2
    assert df_first.loc[df_first["Date"] == "2023-01-01", "Sales"].iloc[0] == 100.0
    
    # Test sum
    df_sum = resolve_duplicates(df, method="sum")
    assert len(df_sum) == 2
    assert df_sum.loc[df_sum["Date"] == "2023-01-01", "Sales"].iloc[0] == 250.0


def test_identify_missing_months_sequence():
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-05-01")
    # Missing March 2023
    existing = pd.Series(pd.to_datetime(["2023-01-01", "2023-02-01", "2023-04-01", "2023-05-01"]))
    missing = identify_missing_months_sequence(start, end, existing)
    assert len(missing) == 1
    assert missing[0] == pd.Timestamp("2023-03-01")


def test_zero_sales_preservation():
    # April 2020 national lockdown case
    df = pd.DataFrame({
        "Date": ["2020-03-01", "2020-04-01", "2020-05-01"],
        "Sales": [140000, 0, 36000]
    })
    clean_df, report = preprocess_car_sales_data(df, allow_zero_sales=True)
    assert len(clean_df) == 3
    assert report.true_zero_sales_count == 1
    assert "2020-04" in report.true_zero_months
    # Verify zero wasn't silently dropped or converted to NaN
    assert (clean_df["Sales"] == 0).any()


def test_imputation_options():
    # Test with gap at 2023-02
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-03-01"],
        "Sales": [100.0, 300.0]
    })
    # None: should keep NaN at missing month
    df_none, report_none = preprocess_car_sales_data(df, imputation_method="none")
    assert len(df_none) == 3
    assert np.isnan(df_none.loc[df_none["Date"] == "2023-02-01", "Sales"].iloc[0])
    
    # Linear interpolation: (100 + 300) / 2 = 200
    df_interp, report_interp = preprocess_car_sales_data(df, imputation_method="linear_interpolation")
    assert df_interp.loc[df_interp["Date"] == "2023-02-01", "Sales"].iloc[0] == 200.0
    
    # Forward fill: should carry forward 100
    df_ffill, report_ffill = preprocess_car_sales_data(df, imputation_method="forward_fill")
    assert df_ffill.loc[df_ffill["Date"] == "2023-02-01", "Sales"].iloc[0] == 100.0


def test_column_mapping_heuristics():
    df_custom = pd.DataFrame({
        "Observation_Period": ["2023-01-01", "2023-02-01"],
        "Vehicle_Volume": [1000, 2000],
        "Category": ["SUV", "Sedan"]
    })
    suggested = detect_suggested_columns(df_custom)
    assert suggested["date_column"] == "Observation_Period"
    assert suggested["sales_column"] == "Vehicle_Volume"
    
    standardized = map_and_standardize_columns(
        df_custom,
        date_col=suggested["date_column"],
        sales_col=suggested["sales_column"]
    )
    assert "Date" in standardized.columns
    assert "Sales" in standardized.columns
    assert "Category" in standardized.columns


def test_invalid_input_handling():
    with pytest.raises(KeyError):
        preprocess_car_sales_data(pd.DataFrame({"Wrong1": [1], "Wrong2": [2]}))
        
    empty_df = pd.DataFrame({"Date": [], "Sales": []})
    with pytest.raises(ValueError):
        preprocess_car_sales_data(empty_df)
