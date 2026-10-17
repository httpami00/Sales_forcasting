# India Car Sales Forecasting and Predictive Analytics System

An end-to-end Data Science and Machine Learning analytics platform for analyzing historical monthly passenger car dispatches in India, discovering cyclical seasonality, and forecasting future automotive demand using statistical and Bayesian time-series techniques (**ARIMA**, **SARIMA**, **Prophet**, and **Seasonal Naive** baselines).

---

## 1. Project Overview & Business Problem

The automotive manufacturing sector operates on extended planning horizons (typically 6 to 18 months) involving complex multi-tier supply chains, raw material procurement (sheet metal, semiconductors, electronic control units, polymers), and capital-intensive tooling. 

In India—the world's third-largest automotive market—demand is highly volatile and seasonal, driven by cultural events (Navratri, Dussehra, Dhanteras, Diwali), macroeconomic shifts (interest rate cycles, fuel prices, inflation), fiscal year-end corporate depreciation pushes (March), and monsoons.

### Objective
Provide automotive executives, supply chain strategists, and dealership planners with an interactive, evidence-based platform to:
1. Ingest, validate, and preprocess historical monthly automotive sales records.
2. Conduct rigorous Exploratory Data Analysis (EDA), seasonal decomposition, and statistical anomaly detection.
3. Compare competing time-series models on strict out-of-sample chronological test windows.
4. Project future demand horizons (3, 6, 12, 18 months ahead) accompanied by 95% confidence bounds.
5. Translate statistical forecasts into actionable automotive production and dealer inventory guidance.

---

## 2. Dataset Attribution, Metric Definitions & Limitations

### Data Source & Attribution
- **Primary Source**: Society of Indian Automobile Manufacturers ([SIAM](https://www.siam.in)) monthly statistical dispatches.
- **Reference / Aggregator**: [Trendonify India Car Sales](https://trendonify.com/india/car-sales), AutoPunditz, and FADA Research.
- **Observation Frequency**: Monthly (`MS` - Month Start).
- **Temporal Coverage**: January 2018 (`2018-01-01`) through December 2024 (`2024-12-01`) (84 continuous months).

### Critical Distinction of Automotive Metrics
In automotive data science, volume metrics differ fundamentally across the distribution chain:
- **Wholesale Dispatches (The Metric Recorded in This System)**: Factory-to-dealer dispatches reported by OEMs through SIAM. This reflects OEM production throughput and inventory buildup ahead of retail demand.
- **Retail Registrations (FADA / Vahan)**: Motor vehicle registrations recorded at Regional Transport Offices (RTOs) upon physical handover to retail buyers.
- **Customer Bookings / Orders**: Forward orders received by dealerships, often subject to cancellation.

> **Zero-Sales Period Context**:
> The `2020-04-01` observation of `0` is an authentic real-world event reflecting the nationwide COVID-19 lockdown when all automotive production and dealer deliveries in India were completely suspended.

### Dataset Limitations
1. **Luxury Automaker Exclusion**: Certain premium OEMs (such as BMW, Mercedes-Benz, JLR, Volvo) do not report dispatches monthly to SIAM and are excluded from standard industry totals.
2. **Aggregated National Volume**: The primary dataset reflects all-India aggregates and does not segment by state, powertrain (EV vs. ICE vs. Hybrid), or vehicle sub-segment (hatchback vs. SUV).
3. **Macroeconomic Shocks**: Events like the April 2020 lockdown, May 2021 COVID wave 2, and September 2021 global semiconductor shortage represent external structural breaks rather than intrinsic seasonal variance.

---

## 3. Technology Stack

- **Runtime**: Python 3.12 (fully tested on Windows 64-bit)
- **Interactive Dashboard**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Visualizations**: Plotly (interactive hover tooltips, multi-traces, shaded confidence bands)
- **Statistical Forecasting**: Statsmodels (`ARIMA`, `SARIMAX`, `seasonal_decompose`)
- **Bayesian Time-Series**: Prophet (additive yearly seasonality)
- **Evaluation**: Scikit-Learn metrics, custom time-series cross-validation
- **Document / Report Generation**: ReportLab (PDF export)
- **Automated Quality Testing**: Pytest

---

## 4. Architecture & Project Structure

```
sales_forcasting/
├── app.py                     # Main multi-page Streamlit application
├── requirements.txt           # Pinned production dependencies
├── README.md                  # Detailed documentation & operational guide
├── .gitignore                 # Git ignore rules
├── data/
│   ├── historical_india_car_sales.csv  # 84-month authentic SIAM dataset (2018–2024)
│   └── README.md              # Data dictionary & provenance
├── sample_data/
│   ├── synthetic_car_sales_demo.csv    # Labeled synthetic dataset for fallback testing
│   ├── sample_promotional_data.csv     # Monthly promotional & festive indicator series
│   └── README.md              # Documentation for sample datasets
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV/Excel loaders, column mapping heuristics, fallback loaders
│   ├── preprocessing.py       # Mixed date parser, duplicate resolution, imputation, zero-handling
│   ├── eda.py                 # Sales KPIs, rolling metrics, calendar seasonality, anomaly detection
│   ├── forecasting.py         # Naive, Seasonal Naive, ARIMA, SARIMA (with exog), Prophet
│   ├── evaluation.py          # Chronological split, MAE/RMSE/WAPE/safe-MAPE, residual analysis
│   ├── insights.py            # Evidence-based automotive market observations & inventory schedule
│   ├── visualizations.py      # Plotly interactive charts & styling
│   └── report_generator.py    # Plain text & ReportLab PDF report generation
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py  # Unit tests for preprocessing, date parsing, imputation
    ├── test_evaluation.py     # Unit tests for metrics, chronological splits, rankings
    └── test_forecasting.py    # Unit tests for all 5 time-series models
```

---

## 5. Installation & Setup Guide (Windows)

### Prerequisites
- Python 3.11 or 3.12 installed on your machine.
- PowerShell or Windows Command Prompt.

### Step 1: Clone or Navigate to the Workspace
```powershell
cd c:\sales_forcasting
```

### Step 2: Create a Virtual Environment
Using Python's standard `venv` or `uv`:
```powershell
# Using Python standard venv
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```
*(If using `uv` for ultra-fast installation: `uv pip install -r requirements.txt`)*

### Step 4: Run the Automated Test Suite
Verify that all unit and integration tests pass:
```powershell
pytest -v
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 6. Time-Series Models Explained

### Model A: Naive Forecast (Flat Persistence)
- **Concept**: Carries the most recent observation forward across the horizon: $\hat{y}_{t+h} = y_t$.
- **Role**: Essential baseline benchmark. Any statistical or ML model must outperform the naive baseline to justify deployment.

### Model B: Seasonal Naive Forecast ($t - 12$)
- **Concept**: Uses the observation from the corresponding month of the prior calendar cycle: $\hat{y}_{t+h} = y_{t+h-12}$.
- **Role**: Primary seasonal baseline. In markets with strong recurring festive patterns, Seasonal Naive often sets an aggressive performance benchmark.

### Model C: ARIMA (Autoregressive Integrated Moving Average)
- **Specification**: $\text{ARIMA}(p, d, q)$
  - $p$: Order of autoregression (lags of the series).
  - $d$: Order of first-differencing to achieve mean-stationarity.
  - $q$: Order of moving average (lags of forecast errors).
- **Role**: Captures short-term auto-correlation and underlying dispatch momentum.

### Model D: SARIMA / SARIMAX (Seasonal ARIMA with Exogenous Regressors)
- **Specification**: $\text{SARIMA}(p, d, q) \times (P, D, Q)_s$ with seasonal period $s = 12$.
  - Models both non-seasonal dynamics and annual 12-month cyclical fluctuations.
  - When promotional datasets are aligned, SARIMAX incorporates exogenous regressors ($X_t$) to isolate promotional marketing uplifts.

### Model E: Prophet (Bayesian Additive Regression)
- **Specification**: Deconstructs time series into trend, periodic seasonality, and irregular components:
  $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
  - $g(t)$: Piecewise linear or logistic growth curve.
  - $s(t)$: Fourier series modeling 12-month seasonality.
  - Generates Bayesian posterior prediction intervals ($\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}$).

---

## 7. Model Evaluation Metrics

Time series models are evaluated strictly on held-out chronological test observations without data leakage:

| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Average dispatch error in absolute physical vehicle units. |
| **RMSE** (Root Mean Squared Error) | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Penalizes large forecasting errors heavily (sensitive to supply shocks). |
| **WAPE** (Weighted Absolute Percentage Error) | $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$ | Normalized aggregate percentage error across the entire test volume. |
| **MAPE** (Safe Mean Absolute Percentage Error) | $\frac{1}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$ | Percentage error per month. Observations where $y_i = 0$ (e.g. April 2020) are filtered out safely to prevent division by zero. |

---

## 8. Dashboard Walkthrough & User Guide

1. **Page 1: Overview & Key Metrics**: High-level KPIs, cumulative historical dispatches, latest MoM/YoY growth, and smoothing moving averages (3M, 6M, 12M).
2. **Page 2: Data Explorer & Mapping**: Upload external CSV/Excel dispatches, map date/sales columns, inspect duplicate months, configure missing-gap imputation, and export cleaned CSV files.
3. **Page 3: Exploratory Analysis (EDA)**: Interactive slicing (full, 5-year, 2-year), calendar month box plots, Year-vs-Month heatmaps, 4-panel seasonal decomposition, and IQR anomaly detection.
4. **Page 4: Time-Series Forecasting**: Select model (SARIMA, ARIMA, Prophet, Naive), configure forecast horizon (3, 6, 12, 18 months), toggle exogenous marketing variables, and review 95% confidence bounds.
5. **Page 5: Model Evaluation & Benchmark**: Compare all models on an identical chronological test window, inspect the Leaderboard, and diagnose residual error distributions.
6. **Page 6: Strategic Business Insights**: Actionable guidance for automotive assembly schedules, pre-festive dealer buffer targets, and promotional factor disclaimers.
7. **Page 7: Export & Executive Reports**: Download structured prediction CSVs, plain text executive intelligence summaries, and formal ReportLab PDF reports.

---

## 9. Dashboard Interface Preview

```
+-----------------------------------------------------------------------------------------+
|  🚗 India Car Sales Forecasting and Predictive Analytics System                        |
+-----------------------------------------------------------------------------------------+
|  [Total Dispatches: 22.4M Units]  [Monthly Avg: 267k]  [Latest: 298k]  [YoY: +4.2%]    |
|-----------------------------------------------------------------------------------------|
|  [Interactive Plotly Time-Series Chart: Actuals + 12M Rolling Trend + Festive Peaks]    |
|                                                                                         |
|  * 2018-2019: Pre-COVID Baseline                                                        |
|  * 2020-04: Lockdown Shutdown Washout (0 units)                                         |
|  * 2021-2024: Post-Pandemic Record Expansion (>4.1M units/year)                         |
+-----------------------------------------------------------------------------------------+
|  [Model Benchmark]   SARIMA: MAE 14,210  |  Prophet: MAE 16,540  |  Naive: MAE 28,100   |
|  [12-Month Forecast] Projected 2025 dispatches with 95% confidence intervals            |
+-----------------------------------------------------------------------------------------+
```

---

## 10. Known Limitations & Future Roadmap

- **Sub-segment Disaggregation**: Future releases can integrate powertrain filters (Battery Electric Vehicles, Strong Hybrids, CNG, Petrol, Diesel) as granular data becomes publicly accessible.
- **Regional / State-Level Breakdown**: Integrating state transport commissioner registrations (Vahan) across regional clusters (North, South, West, East).
- **Macroeconomic Exogenous Pipelines**: Automated API integration for benchmark lending rates (RBI Repo Rate), wholesale fuel prices, and raw steel commodity price indices.

---

## 11. License & Compliance
This software is provided for academic, analytical, and professional research purposes. Data points cited from SIAM, Trendonify, and AutoPunditz remain the intellectual property of their respective reporting agencies. Future forecasts represent mathematical estimates and should not be construed as official manufacturer delivery commitments.
