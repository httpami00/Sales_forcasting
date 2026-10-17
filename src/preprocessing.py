"""
Preprocessing pipeline for India Car Sales Forecasting System.
Handles date parsing, numeric coercion, duplicate detection, missing month identification,
frequency normalization, zero-sales handling, and explicit imputation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class PreprocessingReport:
    """Encapsulates validation and preprocessing diagnostics."""
    def __init__(self):
        self.original_row_count: int = 0
        self.final_row_count: int = 0
        self.duplicate_count: int = 0
        self.duplicate_months: List[str] = []
        self.deduplication_method_used: Optional[str] = None
        self.invalid_date_count: int = 0
        self.invalid_sales_count: int = 0
        self.negative_sales_count: int = 0
        self.true_zero_sales_count: int = 0
        self.true_zero_months: List[str] = []
        self.missing_months_in_sequence: List[str] = []
        self.imputation_method_applied: str = "none"
        self.imputed_months_count: int = 0
        self.start_date: Optional[pd.Timestamp] = None
        self.end_date: Optional[pd.Timestamp] = None
        self.warnings: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_rows": self.original_row_count,
            "final_rows": self.final_row_count,
            "duplicate_count": self.duplicate_count,
            "duplicate_months": self.duplicate_months,
            "invalid_dates": self.invalid_date_count,
            "invalid_sales": self.invalid_sales_count,
            "negative_sales": self.negative_sales_count,
            "true_zero_sales_count": self.true_zero_sales_count,
            "true_zero_months": self.true_zero_months,
            "missing_months_in_sequence": self.missing_months_in_sequence,
            "imputation_method": self.imputation_method_applied,
            "imputed_months_count": self.imputed_months_count,
            "start_date": str(self.start_date.date()) if self.start_date is not None else None,
            "end_date": str(self.end_date.date()) if self.end_date is not None else None,
            "warnings": self.warnings
        }


def clean_numeric_sales(series: pd.Series) -> pd.Series:
    """
    Clean and convert sales values to numeric float/int.
    Removes commas, spaces, currency symbols, and handles strings.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("units", "", case=False, regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_dates_to_month_start(series: pd.Series) -> pd.Series:
    """
    Parse dates flexibly (including mixed formats) and normalize them to Month-Start (1st day of month).
    """
    try:
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    except Exception:
        parsed = pd.to_datetime(series, errors="coerce")
    # Normalize to 1st day of the month
    normalized = parsed.apply(lambda dt: dt.replace(day=1) if pd.notnull(dt) else pd.NaT)
    return normalized


def detect_duplicates(df: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Detect duplicated months in the dataset.
    Returns (duplicate_count, list_of_duplicate_month_strings).
    """
    if "Date" not in df.columns:
        return 0, []
    
    dup_mask = df.duplicated(subset=["Date"], keep=False)
    dup_dates = df.loc[dup_mask, "Date"].dropna().dt.strftime("%Y-%m").unique().tolist()
    dup_count = int(dup_mask.sum())
    return dup_count, dup_dates


def resolve_duplicates(
    df: pd.DataFrame,
    method: str = "keep_first"
) -> pd.DataFrame:
    """
    Resolve duplicate months using specified strategy:
    'keep_first', 'keep_last', 'sum', 'mean'.
    """
    if method == "keep_first":
        return df.drop_duplicates(subset=["Date"], keep="first").copy()
    elif method == "keep_last":
        return df.drop_duplicates(subset=["Date"], keep="last").copy()
    elif method == "sum":
        return df.groupby("Date", as_index=False)["Sales"].sum()
    elif method == "mean":
        return df.groupby("Date", as_index=False)["Sales"].mean()
    else:
        raise ValueError(f"Unknown deduplication method: {method}")


def identify_missing_months_sequence(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    existing_dates: pd.Series
) -> List[pd.Timestamp]:
    """
    Generate expected monthly range ('MS') and identify missing months.
    """
    if pd.isnull(start_date) or pd.isnull(end_date) or start_date >= end_date:
        return []
    
    expected_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    existing_set = set(pd.to_datetime(existing_dates).dropna())
    missing = [d for d in expected_range if d not in existing_set]
    return missing


def preprocess_car_sales_data(
    df: pd.DataFrame,
    dedup_method: str = "keep_first",
    imputation_method: str = "none",
    allow_zero_sales: bool = True
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Execute end-to-end preprocessing pipeline on car sales dataframe.
    
    Parameters:
    - df: Input dataframe with 'Date' and 'Sales' columns
    - dedup_method: 'keep_first', 'keep_last', 'sum', 'mean'
    - imputation_method: 'none', 'forward_fill', 'linear_interpolation', 'seasonal_mean'
    - allow_zero_sales: If True, preserves legitimate zero-sales observations (e.g. April 2020 lockdown)
    
    Returns:
    - (cleaned_df, PreprocessingReport)
    """
    report = PreprocessingReport()
    report.original_row_count = len(df)
    
    if "Date" not in df.columns or "Sales" not in df.columns:
        raise KeyError("Dataset must contain 'Date' and 'Sales' columns before preprocessing.")
    
    working_df = df.copy()
    
    # 1. Parse Date and Sales
    working_df["Date"] = parse_dates_to_month_start(working_df["Date"])
    working_df["Sales"] = clean_numeric_sales(working_df["Sales"])
    
    # Check invalid dates & sales
    report.invalid_date_count = int(working_df["Date"].isna().sum())
    report.invalid_sales_count = int(working_df["Sales"].isna().sum())
    
    if report.invalid_date_count > 0:
        report.warnings.append(f"Dropped {report.invalid_date_count} records with unparseable dates.")
        working_df = working_df.dropna(subset=["Date"])
        
    # 2. Check for negative sales
    neg_mask = working_df["Sales"] < 0
    report.negative_sales_count = int(neg_mask.sum())
    if report.negative_sales_count > 0:
        report.warnings.append(f"Detected {report.negative_sales_count} negative sales values. Negative values set to NaN.")
        working_df.loc[neg_mask, "Sales"] = np.nan
        
    # 3. Check for true zero sales
    zero_mask = working_df["Sales"] == 0
    report.true_zero_sales_count = int(zero_mask.sum())
    if report.true_zero_sales_count > 0:
        zero_dates = working_df.loc[zero_mask, "Date"].dt.strftime("%Y-%m").tolist()
        report.true_zero_months = zero_dates
        report.warnings.append(
            f"Detected {report.true_zero_sales_count} genuine zero-sales observation(s) at: {', '.join(zero_dates)}. "
            "These represent actual real-world shutdowns (e.g. April 2020 national COVID lockdown) and are preserved."
        )
    
    # 4. Detect and report duplicates
    dup_count, dup_months = detect_duplicates(working_df)
    report.duplicate_count = dup_count
    report.duplicate_months = dup_months
    report.deduplication_method_used = dedup_method
    
    if dup_count > 0:
        report.warnings.append(
            f"Detected {dup_count} duplicate month entries ({', '.join(dup_months[:5])}...). "
            f"Resolved using '{dedup_method}' strategy."
        )
        working_df = resolve_duplicates(working_df, method=dedup_method)
        
    # 5. Chronological sort
    working_df = working_df.sort_values("Date").reset_index(drop=True)
    
    if len(working_df) == 0:
        raise ValueError("No valid records remaining after initial date/sales cleaning.")
        
    report.start_date = working_df["Date"].min()
    report.end_date = working_df["Date"].max()
    
    # 6. Identify missing months in chronological sequence
    missing_dates = identify_missing_months_sequence(
        report.start_date, report.end_date, working_df["Date"]
    )
    report.missing_months_in_sequence = [d.strftime("%Y-%m") for d in missing_dates]
    
    if len(missing_dates) > 0:
        report.warnings.append(
            f"Identified {len(missing_dates)} missing calendar month(s) in the historical sequence: "
            f"{', '.join(report.missing_months_in_sequence[:6])}{'...' if len(missing_dates) > 6 else ''}."
        )
        
    # 7. Normalize frequency to continuous Month-Start ('MS')
    working_df = working_df.set_index("Date")
    # Reindex to full expected range to expose missing intervals explicitly
    full_date_idx = pd.date_range(start=report.start_date, end=report.end_date, freq="MS")
    normalized_df = working_df.reindex(full_date_idx)
    normalized_df.index.name = "Date"
    
    # 8. Explicit Imputation Handling
    report.imputation_method_applied = imputation_method
    nan_before_impute = int(normalized_df["Sales"].isna().sum())
    
    if nan_before_impute > 0:
        if imputation_method == "none":
            report.warnings.append(
                f"Dataset contains {nan_before_impute} missing monthly value(s). "
                "Imputation is set to 'None' — models will require complete series or handle missing data as configured."
            )
        elif imputation_method == "forward_fill":
            normalized_df["Sales"] = normalized_df["Sales"].ffill()
            report.imputed_months_count = nan_before_impute
            report.warnings.append(
                f"Imputed {nan_before_impute} month(s) using Forward Fill (previous month's sales carried forward)."
            )
        elif imputation_method == "linear_interpolation":
            normalized_df["Sales"] = normalized_df["Sales"].interpolate(method="linear")
            report.imputed_months_count = nan_before_impute
            report.warnings.append(
                f"Imputed {nan_before_impute} month(s) using Linear Interpolation across adjacent observed periods."
            )
        elif imputation_method == "seasonal_mean":
            # Impute using average for that calendar month (1-12)
            normalized_df["_month"] = normalized_df.index.month
            month_means = normalized_df.groupby("_month")["Sales"].transform("mean")
            normalized_df["Sales"] = normalized_df["Sales"].fillna(month_means)
            normalized_df = normalized_df.drop(columns=["_month"])
            report.imputed_months_count = nan_before_impute
            report.warnings.append(
                f"Imputed {nan_before_impute} month(s) using Seasonal Mean (historical average of the corresponding calendar month)."
            )
            
    final_df = normalized_df.reset_index()
    report.final_row_count = len(final_df)
    
    return final_df, report
