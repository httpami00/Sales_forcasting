"""
Data loader module for India Car Sales Forecasting System.
Handles loading from CSV, Excel, local files, bundled datasets, and column mapping.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


DEFAULT_HISTORICAL_PATH = Path(__file__).resolve().parent.parent / "data" / "historical_india_car_sales.csv"
DEFAULT_DEMO_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "synthetic_car_sales_demo.csv"
DEFAULT_PROMO_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_promotional_data.csv"


def load_raw_dataset(
    file_source: Union[str, Path, io.BytesIO, io.StringIO],
    file_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a raw dataset from a file path or uploaded buffer (CSV or Excel).
    Handles multiple encodings, delimiters, and file formats gracefully.
    """
    name_str = str(file_name or file_source).lower()
    
    if name_str.endswith(".xlsx") or name_str.endswith(".xls"):
        return pd.read_excel(file_source)
    
    # Attempt reading CSV with standard encodings
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    last_err = None
    
    for enc in encodings:
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_csv(file_source, encoding=enc, sep=None, engine="python")
            return df
        except Exception as e:
            last_err = e
            continue
            
    # Fallback to standard comma separator if engine='python' failed
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    try:
        return pd.read_csv(file_source, encoding="utf-8")
    except Exception:
        raise ValueError(f"Unable to read file as CSV or Excel. Details: {last_err}")


def detect_suggested_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Heuristically detect likely Date and Sales columns from a DataFrame.
    """
    cols = list(df.columns)
    date_candidates = ["date", "month", "period", "timestamp", "year_month", "time", "observation_date", "dt"]
    sales_candidates = ["sales", "units", "dispatches", "registrations", "volume", "car_sales", "vehicle_sales", "count", "value", "quantity"]

    detected_date = None
    detected_sales = None

    # Find Date column
    for col in cols:
        clean_col = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if clean_col in date_candidates or any(cand in clean_col for cand in date_candidates):
            detected_date = col
            break
            
    # Find Sales column
    for col in cols:
        clean_col = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if clean_col != detected_date and (clean_col in sales_candidates or any(cand in clean_col for cand in sales_candidates)):
            detected_sales = col
            break

    # Fallbacks if not found by name
    if detected_date is None and len(cols) > 0:
        detected_date = cols[0]
    if detected_sales is None and len(cols) > 1:
        detected_sales = cols[1]

    return {
        "date_column": detected_date,
        "sales_column": detected_sales
    }


def map_and_standardize_columns(
    df: pd.DataFrame,
    date_col: str,
    sales_col: str
) -> pd.DataFrame:
    """
    Standardize the selected date and sales columns to 'Date' and 'Sales'.
    Preserves other original columns.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in dataset columns: {list(df.columns)}")
    if sales_col not in df.columns:
        raise KeyError(f"Sales column '{sales_col}' not found in dataset columns: {list(df.columns)}")

    df_clean = df.copy()
    
    # Store original reference columns before renaming
    df_clean["_original_date_raw"] = df_clean[date_col]
    df_clean["_original_sales_raw"] = df_clean[sales_col]
    
    # Standardize column names
    rename_dict = {date_col: "Date", sales_col: "Sales"}
    # If date_col or sales_col was already named Date/Sales, rename safely
    if date_col != "Date" and "Date" in df_clean.columns:
        df_clean = df_clean.rename(columns={"Date": "Date_orig_col"})
    if sales_col != "Sales" and "Sales" in df_clean.columns:
        df_clean = df_clean.rename(columns={"Sales": "Sales_orig_col"})
        
    df_clean = df_clean.rename(columns=rename_dict)
    return df_clean


def load_bundled_historical_sales() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load the bundled authentic historical Indian Car Sales dataset (SIAM / Industry data).
    Returns (DataFrame, metadata_dict).
    """
    if not DEFAULT_HISTORICAL_PATH.exists():
        raise FileNotFoundError(f"Bundled historical file not found at {DEFAULT_HISTORICAL_PATH}")
    
    df = pd.read_csv(DEFAULT_HISTORICAL_PATH)
    meta = {
        "dataset_name": "Indian Passenger Vehicle Monthly Dispatches (SIAM / Industry Historical)",
        "source": "Society of Indian Automobile Manufacturers (SIAM) / Trendonify Benchmark Reference",
        "is_synthetic": False,
        "description": "Authentic monthly domestic car dispatches across India from January 2018 to December 2024.",
        "path": str(DEFAULT_HISTORICAL_PATH)
    }
    return df, meta


def load_synthetic_demo_sales() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load the bundled labeled synthetic demo dataset.
    Returns (DataFrame, metadata_dict).
    """
    if not DEFAULT_DEMO_PATH.exists():
        raise FileNotFoundError(f"Synthetic demo file not found at {DEFAULT_DEMO_PATH}")
    
    df = pd.read_csv(DEFAULT_DEMO_PATH)
    meta = {
        "dataset_name": "SYNTHETIC DEMO - India Car Sales Simulation",
        "source": "Synthetic Benchmark Generator (Explicitly Labeled Demo)",
        "is_synthetic": True,
        "description": "Simulated car sales data with realistic seasonality and COVID shock for demo and testing.",
        "path": str(DEFAULT_DEMO_PATH)
    }
    return df, meta


def load_promotional_dataset(
    file_source: Optional[Union[str, Path, io.BytesIO]] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load the promotional and festive indicators dataset.
    Uses default bundled promotional dataset if no external file provided.
    """
    if file_source is None:
        if not DEFAULT_PROMO_PATH.exists():
            raise FileNotFoundError(f"Default promotional file not found at {DEFAULT_PROMO_PATH}")
        df = pd.read_csv(DEFAULT_PROMO_PATH)
        source_label = "Bundled India Automotive Promotional Index (Sample Data)"
    else:
        df = load_raw_dataset(file_source)
        source_label = "User Uploaded Promotional Dataset"

    meta = {
        "source": source_label,
        "columns": list(df.columns)
    }
    return df, meta


def get_data_acquisition_guide() -> str:
    """
    Return step-by-step guidance on obtaining authentic Indian car sales data.
    """
    return """
### How to Obtain Authentic Indian Automotive Sales Data:
1. **Primary Source - SIAM (Society of Indian Automobile Manufacturers)**:
   - Visit [SIAM Official Portal](https://www.siam.in).
   - Under *Industry Statistics*, monthly releases detail Passenger Vehicles (Cars, Utility Vehicles, Vans) domestic dispatches.
2. **Retail Registration Data - FADA (Federation of Automobile Dealers Associations)**:
   - Monthly vehicle retail registration data sourced from MoRTH Vahan dashboard.
   - Published monthly around the 1st week of each calendar month.
3. **Reference Portal - Trendonify**:
   - [Trendonify India Car Sales](https://trendonify.com/india/car-sales) aggregates financial and automotive historical dispatches.
   - Download the monthly table as CSV or Excel and upload it using the Data Explorer page.
"""
