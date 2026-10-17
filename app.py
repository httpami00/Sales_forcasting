"""
India Car Sales Forecasting and Predictive Analytics System
Interactive Streamlit Dashboard
"""

import os
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="India Car Sales Forecasting & Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from src.data_loader import (
    load_raw_dataset,
    detect_suggested_columns,
    map_and_standardize_columns,
    load_bundled_historical_sales,
    load_synthetic_demo_sales,
    load_promotional_dataset,
    get_data_acquisition_guide
)
from src.preprocessing import (
    preprocess_car_sales_data,
    PreprocessingReport
)
from src.eda import (
    calculate_sales_kpis,
    compute_growth_and_rolling_metrics,
    get_recent_slices,
    analyze_monthly_seasonality,
    perform_seasonal_decomposition,
    detect_statistical_anomalies
)
from src.forecasting import (
    forecast_naive,
    forecast_seasonal_naive,
    forecast_arima,
    forecast_sarima,
    forecast_prophet,
    run_all_available_models,
    generate_future_forecast_dates,
    ForecastResult,
    PROPHET_AVAILABLE,
    PROPHET_ERROR_MSG
)
from src.evaluation import (
    split_time_series_chronologically,
    calculate_evaluation_metrics,
    compare_and_rank_models,
    compute_residual_diagnostics,
    perform_expanding_window_cv
)
from src.insights import generate_automotive_business_insights
from src.visualizations import (
    plot_historical_sales_and_rolling,
    plot_anomaly_detection_scatter,
    plot_seasonality_month_bars,
    plot_seasonality_heatmap,
    plot_seasonal_decomposition_components,
    plot_test_evaluation_comparison,
    plot_future_forecast_chart,
    plot_residual_diagnostics_charts
)
from src.report_generator import (
    generate_text_executive_report,
    generate_pdf_report_bytes,
    REPORTLAB_AVAILABLE
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.06);
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #0F172A;
    }
    .badge-primary {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .badge-synthetic {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #FDE68A;
    }
    .alert-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .alert-info {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "raw_df" not in st.session_state:
    # Auto-load bundled authentic SIAM historical data as default
    try:
        df_hist, meta_hist = load_bundled_historical_sales()
        st.session_state.raw_df = df_hist
        st.session_state.dataset_meta = meta_hist
        st.session_state.date_col = "Date"
        st.session_state.sales_col = "Sales"
        clean_df, report = preprocess_car_sales_data(df_hist, dedup_method="keep_first", imputation_method="none")
        st.session_state.clean_df = clean_df
        st.session_state.prep_report = report
    except Exception as e:
        st.session_state.raw_df = None
        st.session_state.clean_df = None
        st.session_state.dataset_meta = {"dataset_name": "No data loaded", "is_synthetic": False}

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

if "latest_forecast_result" not in st.session_state:
    st.session_state.latest_forecast_result = None

if "promotional_df" not in st.session_state:
    try:
        promo_df, _ = load_promotional_dataset()
        st.session_state.promotional_df = promo_df
    except Exception:
        st.session_state.promotional_df = None


# ==============================================================================
# Sidebar Navigation & Controls
# ==============================================================================
st.sidebar.image("https://img.icons8.com/isometric/100/car.png", width=64)
st.sidebar.title("India Car Sales")
st.sidebar.caption("Forecasting & Predictive Analytics System")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Overview & Key Metrics",
        "2. Data Explorer & Mapping",
        "3. Exploratory Analysis (EDA)",
        "4. Time-Series Forecasting",
        "5. Model Evaluation & Benchmark",
        "6. Strategic Business Insights",
        "7. Export & Executive Reports"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Active Dataset Status")
meta = st.session_state.get("dataset_meta", {})
if st.session_state.clean_df is not None:
    st.sidebar.markdown(f"**Dataset**: {meta.get('dataset_name', 'Custom')}")
    if meta.get("is_synthetic", False):
        st.sidebar.markdown('<span class="badge-synthetic">SYNTHETIC DEMO</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge-primary">AUTHENTIC SIAM DATA</span>', unsafe_allow_html=True)
    st.sidebar.write(f"Observations: **{len(st.session_state.clean_df)} months**")
    st.sidebar.write(f"Range: **{st.session_state.clean_df['Date'].min().strftime('%b %Y')}** to **{st.session_state.clean_df['Date'].max().strftime('%b %Y')}**")
else:
    st.sidebar.warning("No dataset currently loaded.")

st.sidebar.caption("Sourced from SIAM Industry Dispatches | Reference: Trendonify")


# ==============================================================================
# Helper Function: Ensure Data Exists
# ==============================================================================
def require_clean_data() -> bool:
    if st.session_state.clean_df is None or len(st.session_state.clean_df) == 0:
        st.error("No valid dataset loaded. Please navigate to '2. Data Explorer & Mapping' to load the historical dataset or upload a file.")
        return False
    return True


# ==============================================================================
# PAGE 1: OVERVIEW & KEY METRICS
# ==============================================================================
if page == "1. Overview & Key Metrics":
    st.title("🚗 India Car Sales Forecasting & Predictive Analytics")
    st.markdown("""
    A comprehensive data science system analyzing historical monthly passenger vehicle factory dispatches 
    across the Indian automotive market, identifying demand seasonality, and projecting future demand using 
    time-series forecasting models (**ARIMA**, **SARIMA**, **Prophet**, and **Seasonal Naive** baselines).
    """)
    
    if require_clean_data():
        df = st.session_state.clean_df
        kpis = calculate_sales_kpis(df)
        
        # Attribution & Dataset Banner
        c_badge, c_src = st.columns([1, 4])
        with c_badge:
            if meta.get("is_synthetic", False):
                st.markdown('<span class="badge-synthetic">SYNTHETIC DEMO DATASET</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-primary">OFFICIAL SIAM BENCHMARK DATA</span>', unsafe_allow_html=True)
        with c_src:
            st.caption(f"**Source**: {meta.get('source', 'SIAM')} | Coverage: {kpis.get('start_month')} – {kpis.get('end_month')} ({kpis.get('total_months_count')} valid monthly dispatches)")
            
        st.divider()
        
        # Key Performance Indicator Cards
        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Historical Dispatches",
                value=f"{kpis.get('total_historical_sales', 0):,.0f} units",
                help="Sum of all passenger vehicles dispatched across the entire observation period"
            )
            
        with col2:
            st.metric(
                label="Average Monthly Sales",
                value=f"{kpis.get('average_monthly_sales', 0):,.0f} units",
                help="Mean monthly domestic dispatch across historical observations"
            )
            
        with col3:
            latest_mom = kpis.get("latest_mom_change_pct")
            mom_str = f"{latest_mom:+.1f}%" if latest_mom is not None else "N/A"
            st.metric(
                label=f"Latest Sales ({kpis.get('latest_recorded_month')})",
                value=f"{kpis.get('latest_recorded_sales', 0):,.0f} units",
                delta=f"MoM: {mom_str}",
                help="Latest recorded monthly volume and month-over-month growth"
            )
            
        with col4:
            latest_yoy = kpis.get("latest_yoy_change_pct")
            yoy_str = f"{latest_yoy:+.1f}%" if latest_yoy is not None else "N/A"
            st.metric(
                label="Year-over-Year Growth (Latest)",
                value=yoy_str,
                delta=yoy_str,
                help="Percentage change compared to the exact same calendar month of the previous year"
            )

        col5, col6 = st.columns(2)
        with col5:
            st.info(f"**Highest Recorded Month**: **{kpis.get('highest_sales_month')}** with **{kpis.get('highest_sales_value', 0):,.0f} units** (Festive Season Peak)")
        with col6:
            st.warning(f"**Lowest Recorded Month**: **{kpis.get('lowest_sales_month')}** with **{kpis.get('lowest_sales_value', 0):,.0f} units** (National COVID Lockdown Shutdown)")

        st.divider()
        
        # Primary Trend Chart
        st.subheader("Historical Monthly Dispatches & Smoothing Trends")
        df_growth = compute_growth_and_rolling_metrics(df)
        
        c_roll1, c_roll2, c_roll3 = st.columns(3)
        with c_roll1:
            show_3m = st.checkbox("Show 3-Month Rolling Average", value=True)
        with c_roll2:
            show_6m = st.checkbox("Show 6-Month Rolling Average", value=False)
        with c_roll3:
            show_12m = st.checkbox("Show 12-Month Rolling Trend", value=True)
            
        fig_trend = plot_historical_sales_and_rolling(df_growth, show_3m, show_6m, show_12m)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Quick Insights Summary
        st.subheader("Macro Industry Context")
        st.markdown("""
        - **Pre-COVID Baseline (2018–2019)**: Indian automotive dispatches averaged 250k–300k units/month before encountering a cyclical liquidity slowdown in late 2019.
        - **COVID-19 Washout (April 2020)**: Strict national lockdown halted assembly lines, resulting in 0 factory dispatches.
        - **Post-Pandemic Structural Rebound (2021–2024)**: Rapid recovery driven by personal mobility preference, utility vehicle (SUV) demand surge, and festive volume breakthroughs crossing 390k+ units/month.
        """)


# ==============================================================================
# PAGE 2: DATA EXPLORER & MAPPING
# ==============================================================================
elif page == "2. Data Explorer & Mapping":
    st.title("📂 Data Explorer, Upload & Preprocessing")
    st.markdown("Inspect raw observations, map custom columns, configure deduplication, and handle missing intervals.")
    
    # Dataset Selection & Loading
    st.subheader("1. Data Ingestion Source")
    source_choice = st.radio(
        "Choose Data Source:",
        [
            "Bundled Authentic Historical Dataset (SIAM / Industry 2018–2024)",
            "Upload Custom File (CSV or Excel)",
            "Synthetic Demo Dataset (Fallback Testing)"
        ],
        horizontal=True
    )
    
    if source_choice == "Upload Custom File (CSV or Excel)":
        uploaded_file = st.file_uploader("Upload CSV or Excel file:", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                raw_df = load_raw_dataset(uploaded_file, uploaded_file.name)
                st.session_state.raw_df = raw_df
                st.session_state.dataset_meta = {
                    "dataset_name": uploaded_file.name,
                    "source": "User Uploaded File",
                    "is_synthetic": False
                }
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(raw_df)} rows).")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
    elif source_choice == "Synthetic Demo Dataset (Fallback Testing)":
        if st.button("Load Labeled Synthetic Demo Data"):
            df_demo, meta_demo = load_synthetic_demo_sales()
            st.session_state.raw_df = df_demo
            st.session_state.dataset_meta = meta_demo
            st.info("Loaded synthetic demo dataset for testing.")
    else:
        if st.button("Reload Bundled Authentic SIAM Data"):
            df_hist, meta_hist = load_bundled_historical_sales()
            st.session_state.raw_df = df_hist
            st.session_state.dataset_meta = meta_hist
            st.success("Loaded bundled authentic SIAM historical dataset (84 months).")

    if st.session_state.raw_df is not None:
        raw_df = st.session_state.raw_df
        st.divider()
        
        # Column Mapping
        st.subheader("2. Column Mapping & Schema Standardization")
        suggested = detect_suggested_columns(raw_df)
        cols = list(raw_df.columns)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            date_idx = cols.index(suggested["date_column"]) if suggested["date_column"] in cols else 0
            date_selected = st.selectbox("Select Date / Month Column:", cols, index=date_idx)
        with col_m2:
            sales_idx = cols.index(suggested["sales_column"]) if suggested["sales_column"] in cols else (1 if len(cols) > 1 else 0)
            sales_selected = st.selectbox("Select Sales / Dispatches Column:", cols, index=sales_idx)
            
        st.session_state.date_col = date_selected
        st.session_state.sales_col = sales_selected
        
        standardized_df = map_and_standardize_columns(raw_df, date_selected, sales_selected)
        
        # Preprocessing Configuration Options
        st.divider()
        st.subheader("3. Preprocessing & Data Cleaning Configuration")
        c_prep1, c_prep2, c_prep3 = st.columns(3)
        
        with c_prep1:
            dedup_method = st.selectbox(
                "Duplicate Month Handling:",
                ["keep_first", "keep_last", "sum", "mean"],
                help="Strategy for resolving multiple observations for the same calendar month"
            )
        with c_prep2:
            imputation_method = st.selectbox(
                "Missing Interval Imputation:",
                ["none", "linear_interpolation", "forward_fill", "seasonal_mean"],
                help="Explicit imputation method for gaps in the monthly time sequence"
            )
        with c_prep3:
            preserve_zeros = st.checkbox(
                "Preserve Legitimate Zero Sales",
                value=True,
                help="Distinguish genuine shutdowns (April 2020 national lockdown) from missing entries"
            )
            
        if st.button("Apply Preprocessing Pipeline", type="primary"):
            try:
                clean_df, report = preprocess_car_sales_data(
                    standardized_df,
                    dedup_method=dedup_method,
                    imputation_method=imputation_method,
                    allow_zero_sales=preserve_zeros
                )
                st.session_state.clean_df = clean_df
                st.session_state.prep_report = report
                st.success("Preprocessing pipeline completed successfully!")
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                
        # Preprocessing Report Diagnostics
        if "prep_report" in st.session_state and st.session_state.prep_report is not None:
            rep = st.session_state.prep_report
            st.divider()
            st.subheader("4. Preprocessing Quality Diagnostics")
            
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Original Rows", rep.original_row_count)
            d2.metric("Cleaned Valid Months", rep.final_row_count)
            d3.metric("Duplicates Identified", rep.duplicate_count)
            d4.metric("True Zero Observations", rep.true_zero_sales_count)
            
            if rep.warnings:
                with st.expander("⚠️ View Preprocessing Warnings & Action Log", expanded=True):
                    for w in rep.warnings:
                        st.write(f"- {w}")
                        
            # Data Preview & Download
            st.divider()
            st.subheader("5. Cleaned Dataset Preview")
            st.dataframe(st.session_state.clean_df, use_container_width=True)
            
            csv_data = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_india_car_sales.csv",
                mime="text/csv"
            )
            
    # Data Acquisition Guide Expander
    with st.expander("ℹ️ How to Obtain Authentic Historical Car Sales Data (SIAM / Trendonify)"):
        st.markdown(get_data_acquisition_guide())


# ==============================================================================
# PAGE 3: EXPLORATORY ANALYSIS (EDA)
# ==============================================================================
elif page == "3. Exploratory Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis & Seasonality")
    st.markdown("Discover underlying trends, cyclical month-of-year dynamics, calendar heatmaps, and statistical anomaly detection.")
    
    if require_clean_data():
        df = st.session_state.clean_df
        
        tab_trends, tab_seasonality, tab_decomp, tab_anomalies = st.tabs([
            "📈 Trends & Rolling Growth",
            "📅 Calendar Seasonality",
            "🔬 Seasonal Decomposition",
            "🚨 Anomaly Detection"
        ])
        
        # TAB 1: Trends & Rolling Growth
        with tab_trends:
            st.subheader("Historical Trends & Time-Slice Analysis")
            slice_choice = st.radio("Select Historical Window:", ["Full Dataset", "Recent 5 Years", "Recent 2 Years"], horizontal=True)
            slices = get_recent_slices(df)
            
            if slice_choice == "Recent 5 Years":
                active_slice = slices["5_year"]
            elif slice_choice == "Recent 2 Years":
                active_slice = slices["2_year"]
            else:
                active_slice = slices["full"]
                
            growth_df = compute_growth_and_rolling_metrics(active_slice)
            fig_slice = plot_historical_sales_and_rolling(growth_df, True, True, True)
            st.plotly_chart(fig_slice, use_container_width=True)
            
            # Growth Rates Sub-section
            st.subheader("Month-over-Month & Year-over-Year Growth Trajectory")
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.write("**Month-over-Month (MoM) Growth (%)**")
                st.line_chart(growth_df.set_index("Date")["MoM_Growth_Pct"])
            with c_g2:
                st.write("**Year-over-Year (YoY) Growth (%) (12-Month Base)**")
                st.line_chart(growth_df.set_index("Date")["YoY_Growth_Pct"])

        # TAB 2: Calendar Seasonality
        with tab_seasonality:
            st.subheader("Calendar Month Seasonality Patterns")
            st.markdown("""
            In the Indian automotive market, monthly dispatches exhibit strong structural seasonality:
            - **September – October**: Peak dispatches aligned with Navratri, Dussehra, and Diwali festive buying.
            - **March**: Commercial and corporate depreciation purchases prior to fiscal year closing.
            - **December**: Model-year transition clearance with high promotional dealer incentives.
            - **May – June**: Traditional monsoon and post-fiscal consolidation lull.
            """)
            
            monthly_agg, pivot_heatmap = analyze_monthly_seasonality(df)
            
            col_bar, col_heat = st.columns([1, 1])
            with col_bar:
                fig_bars = plot_seasonality_month_bars(monthly_agg)
                st.plotly_chart(fig_bars, use_container_width=True)
            with col_heat:
                fig_heat = plot_seasonality_heatmap(pivot_heatmap)
                st.plotly_chart(fig_heat, use_container_width=True)
                
            st.dataframe(monthly_agg.rename(columns={"Mean": "Avg_Sales", "Count": "Observed_Years"}), use_container_width=True)

        # TAB 3: Seasonal Decomposition
        with tab_decomp:
            st.subheader("Classical Seasonal Decomposition (12-Month Period)")
            st.markdown("""
            Decomposes the passenger vehicle sales time series into:
            1. **Observed**: Raw historical observations
            2. **Trend**: Long-term moving average trajectory
            3. **Seasonal**: Systematic 12-month cyclical pattern
            4. **Residual**: Irregular fluctuations, external shocks, and noise
            """)
            
            decomp_model = st.radio("Decomposition Model:", ["additive", "multiplicative"], horizontal=True)
            decomp, err = perform_seasonal_decomposition(df, model=decomp_model)
            
            if err:
                st.warning(err)
            elif decomp is not None:
                fig_decomp = plot_seasonal_decomposition_components(decomp)
                st.plotly_chart(fig_decomp, use_container_width=True)
                st.caption("Statistical Note: Recurring patterns reflect seasonal correlation and consumer purchasing cycles, but do not imply strict deterministic causation.")

        # TAB 4: Anomaly Detection
        with tab_anomalies:
            st.subheader("Automated Statistical Anomaly Detection")
            st.markdown("""
            Identifies months with dispatches deviating significantly from expected baseline levels 
            (e.g., national lockdowns, post-lockdown personal mobility spikes, or semiconductor supply bottlenecks).
            **Policy**: Anomalous observations are flagged for visual inspection and risk modeling; **they are not automatically deleted**.
            """)
            
            c_an1, c_an2 = st.columns(2)
            with c_an1:
                anom_method = st.selectbox("Detection Method:", ["iqr", "zscore"])
            with c_an2:
                anom_thresh = st.slider("Sensitivity Threshold:", min_value=1.0, max_value=3.5, value=1.5 if anom_method == "iqr" else 2.5, step=0.1)
                
            df_anomalies = detect_statistical_anomalies(df, method=anom_method, threshold=anom_thresh)
            fig_anom = plot_anomaly_detection_scatter(df_anomalies)
            st.plotly_chart(fig_anom, use_container_width=True)
            
            anom_count = int(df_anomalies["Is_Anomaly"].sum())
            st.write(f"Identified **{anom_count}** anomalous observation(s) across the dataset:")
            if anom_count > 0:
                table_anom = df_anomalies[df_anomalies["Is_Anomaly"]][["Date", "Sales", "Anomaly_Direction", "Anomaly_Reason"]]
                st.dataframe(table_anom, use_container_width=True)


# ==============================================================================
# PAGE 4: TIME-SERIES FORECASTING
# ==============================================================================
elif page == "4. Time-Series Forecasting":
    st.title("🔮 Time-Series Forecasting")
    st.markdown("Generate future monthly passenger vehicle dispatches with 95% confidence intervals.")
    
    if require_clean_data():
        df = st.session_state.clean_df
        clean_series = df.set_index("Date")["Sales"].dropna()
        
        # Forecasting Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            model_options = ["SARIMA (Seasonal ARIMA)", "ARIMA", "Prophet (Additive)", "Seasonal Naive (t-12)", "Naive (Last Value)"]
            selected_model_name = st.selectbox("Select Forecasting Model:", model_options)
            
        with col_ctrl2:
            horizon_steps = st.selectbox("Forecast Horizon:", [3, 6, 12, 18], index=2, format_func=lambda h: f"{h} Months Ahead")
            
        with col_ctrl3:
            fit_mode = st.radio("Training Scope:", ["Fit on Entire Historical Dataset", "Use Chronological Train Split Only"], index=0)

        # Advanced Model Hyperparameters
        with st.expander("⚙️ Advanced Model Hyperparameters & Exogenous Regressors", expanded=False):
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                p_order = st.number_input("ARIMA / SARIMA p (AR order):", min_value=0, max_value=5, value=1)
                d_order = st.number_input("ARIMA / SARIMA d (Differencing):", min_value=0, max_value=2, value=1)
                q_order = st.number_input("ARIMA / SARIMA q (MA order):", min_value=0, max_value=5, value=1)
            with c_p2:
                P_s = st.number_input("SARIMA P (Seasonal AR):", min_value=0, max_value=3, value=1)
                D_s = st.number_input("SARIMA D (Seasonal Diff):", min_value=0, max_value=2, value=1)
                Q_s = st.number_input("SARIMA Q (Seasonal MA):", min_value=0, max_value=3, value=1)
                
            use_exog = st.checkbox("Include Promotional / Festive Exogenous Regressors (SARIMAX)", value=False)
            if use_exog and st.session_state.promotional_df is not None:
                st.caption("Using bundled Indian automotive promotional and festive season indicators.")
            elif use_exog:
                st.warning("Promotional impact cannot be estimated from the current dataset because promotional information is unavailable.")

        if st.button("🚀 Run Forecast", type="primary"):
            with st.spinner(f"Fitting {selected_model_name} and generating {horizon_steps}-month forecast..."):
                last_hist_date = clean_series.index[-1]
                future_dates = generate_future_forecast_dates(last_hist_date, steps=horizon_steps)
                
                fc_result: Optional[ForecastResult] = None
                fc_err = None
                
                try:
                    if selected_model_name == "Naive (Last Value)":
                        fc_result = forecast_naive(clean_series, steps=horizon_steps, future_dates=future_dates)
                    elif selected_model_name == "Seasonal Naive (t-12)":
                        fc_result = forecast_seasonal_naive(clean_series, steps=horizon_steps, future_dates=future_dates)
                    elif selected_model_name == "ARIMA":
                        fc_result = forecast_arima(clean_series, steps=horizon_steps, order=(p_order, d_order, q_order), future_dates=future_dates)
                    elif selected_model_name.startswith("SARIMA"):
                        exog_tr = None
                        exog_fu = None
                        if use_exog and st.session_state.promotional_df is not None:
                            p_df = st.session_state.promotional_df.copy()
                            p_df["Date"] = pd.to_datetime(p_df["Date"])
                            p_df = p_df.set_index("Date")
                            common_cols = [c for c in ["Promo_Index", "Festive_Season"] if c in p_df.columns]
                            if common_cols:
                                exog_tr = p_df.reindex(clean_series.index)[common_cols].fillna(0)
                                exog_fu = p_df.reindex(future_dates)[common_cols].fillna(0)
                        fc_result = forecast_sarima(
                            clean_series,
                            steps=horizon_steps,
                            order=(p_order, d_order, q_order),
                            seasonal_order=(P_s, D_s, Q_s, 12),
                            exog_train=exog_tr,
                            exog_future=exog_fu,
                            future_dates=future_dates
                        )
                    elif selected_model_name.startswith("Prophet"):
                        fc_result = forecast_prophet(clean_series, steps=horizon_steps, future_dates=future_dates)
                except Exception as e:
                    fc_err = str(e)
                    
                if fc_err:
                    st.error(f"Forecasting error: {fc_err}")
                elif fc_result is not None:
                    st.session_state.latest_forecast_result = fc_result
                    st.success(f"Generated {horizon_steps}-month forecast using {fc_result.model_name}!")
                    
        # Render Forecast Output if available
        if st.session_state.latest_forecast_result is not None:
            fc_res = st.session_state.latest_forecast_result
            last_date = clean_series.index[-1]
            last_val = float(clean_series.iloc[-1])
            df_fc = fc_res.to_dataframe(last_observed_date=last_date, last_observed_value=last_val, historical_df=df)
            
            st.divider()
            st.subheader(f"Forecast Projections: {fc_res.model_name}")
            st.caption("⚠️ **Notice**: Future values represent statistical model estimates based on historical patterns, not official industry quotas.")
            
            fig_fc = plot_future_forecast_chart(df, None, df_fc, fc_res.model_name, show_historical_months=24)
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Forecast Table
            st.write("**Monthly Forecast Breakdown**")
            st.dataframe(
                df_fc.style.format({
                    "Forecast": "{:,.0f}",
                    "Lower_Bound": lambda v: f"{v:,.0f}" if pd.notnull(v) else "N/A",
                    "Upper_Bound": lambda v: f"{v:,.0f}" if pd.notnull(v) else "N/A",
                    "MoM_Change_Pct": lambda v: f"{v:+.2f}%" if pd.notnull(v) else "N/A",
                    "YoY_Change_Pct": lambda v: f"{v:+.2f}%" if pd.notnull(v) else "N/A"
                }),
                use_container_width=True
            )
            
            # CSV Download
            csv_fc = df_fc.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_fc,
                file_name=f"india_car_sales_forecast_{fc_res.model_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )


# ==============================================================================
# PAGE 5: MODEL EVALUATION & BENCHMARK
# ==============================================================================
elif page == "5. Model Evaluation & Benchmark":
    st.title("🎯 Model Evaluation & Out-of-Sample Benchmark")
    st.markdown("""
    Evaluate models strictly on **unseen chronological test observations**.
    Never shuffle time-series data. Standard metrics: **MAE**, **RMSE**, **WAPE**, and safe **MAPE**.
    """)
    
    if require_clean_data():
        df = st.session_state.clean_df
        
        c_ev1, c_ev2, c_ev3 = st.columns(3)
        with c_ev1:
            test_months = st.selectbox("Held-out Test Period Size:", [6, 12, 18], index=1, format_func=lambda m: f"Last {m} Months")
        with c_ev2:
            rank_metric = st.selectbox("Rank Benchmark By:", ["MAE", "RMSE", "WAPE_Pct", "MAPE_Pct"])
        with c_ev3:
            run_btn = st.button("🏁 Run Full Model Benchmark", type="primary")

        train_df, test_df, split_warn = split_time_series_chronologically(df, test_size=test_months)
        if split_warn:
            st.info(f"Split Note: {split_warn}")
            
        st.write(f"**Training Set**: {train_df['Date'].min().strftime('%b %Y')} to {train_df['Date'].max().strftime('%b %Y')} ({len(train_df)} months)")
        st.write(f"**Held-out Test Set**: {test_df['Date'].min().strftime('%b %Y')} to {test_df['Date'].max().strftime('%b %Y')} ({len(test_df)} months)")

        if run_btn or st.session_state.eval_results is None:
            with st.spinner("Training models on historical training slice and testing against held-out observations..."):
                train_s = train_df.set_index("Date")["Sales"]
                test_actuals = test_df["Sales"].values
                test_dates = test_df["Date"].values
                
                preds_dict: Dict[str, np.ndarray] = {}
                metrics_list = []
                
                # 1. Naive
                try:
                    res_naive = forecast_naive(train_s, steps=len(test_df), future_dates=test_dates)
                    preds_dict["Naive"] = res_naive.point_forecast
                    metrics_list.append(calculate_evaluation_metrics(test_actuals, res_naive.point_forecast, "Naive"))
                except Exception:
                    pass
                    
                # 2. Seasonal Naive
                try:
                    res_snaive = forecast_seasonal_naive(train_s, steps=len(test_df), future_dates=test_dates)
                    preds_dict["Seasonal Naive"] = res_snaive.point_forecast
                    metrics_list.append(calculate_evaluation_metrics(test_actuals, res_snaive.point_forecast, "Seasonal Naive (t-12)"))
                except Exception:
                    pass
                    
                # 3. ARIMA(1,1,1)
                try:
                    res_arima = forecast_arima(train_s, steps=len(test_df), order=(1, 1, 1), future_dates=test_dates)
                    preds_dict["ARIMA(1,1,1)"] = res_arima.point_forecast
                    metrics_list.append(calculate_evaluation_metrics(test_actuals, res_arima.point_forecast, "ARIMA(1,1,1)"))
                except Exception:
                    pass
                    
                # 4. SARIMA(1,1,1)x(1,1,1,12)
                try:
                    res_sarima = forecast_sarima(train_s, steps=len(test_df), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), future_dates=test_dates)
                    preds_dict["SARIMA(1,1,1)x(1,1,1,12)"] = res_sarima.point_forecast
                    metrics_list.append(calculate_evaluation_metrics(test_actuals, res_sarima.point_forecast, "SARIMA"))
                except Exception:
                    pass
                    
                # 5. Prophet
                if PROPHET_AVAILABLE:
                    try:
                        res_prophet = forecast_prophet(train_s, steps=len(test_df), future_dates=test_dates)
                        preds_dict["Prophet"] = res_prophet.point_forecast
                        metrics_list.append(calculate_evaluation_metrics(test_actuals, res_prophet.point_forecast, "Prophet"))
                    except Exception:
                        pass
                        
                ranked_df = compare_and_rank_models(metrics_list, sort_by=rank_metric)
                st.session_state.eval_results = {
                    "ranked_df": ranked_df,
                    "preds_dict": preds_dict,
                    "train_df": train_df,
                    "test_df": test_df
                }

        # Render Benchmark Results
        if st.session_state.eval_results is not None:
            ev = st.session_state.eval_results
            ranked_df = ev["ranked_df"]
            preds_dict = ev["preds_dict"]
            
            st.divider()
            st.subheader(f"🏆 Model Benchmark Leaderboard (Ranked by {rank_metric})")
            
            best_model = ranked_df.iloc[0]["Model"]
            best_score = ranked_df.iloc[0][rank_metric]
            st.success(f"🥇 **Best Performing Model**: **{best_model}** with {rank_metric} = **{best_score:,.2f}**")
            
            st.dataframe(
                ranked_df.style.format({
                    "MAE": "{:,.0f} units",
                    "RMSE": "{:,.0f} units",
                    "WAPE_Pct": "{:.2f}%",
                    "MAPE_Pct": "{:.2f}%"
                }),
                use_container_width=True
            )
            
            # Actual vs Predicted Visual Comparison
            st.divider()
            st.subheader("Held-Out Test Set: Actual vs Predicted Trajectory")
            fig_test = plot_test_evaluation_comparison(ev["train_df"], ev["test_df"], preds_dict)
            st.plotly_chart(fig_test, use_container_width=True)
            
            # Residual Diagnostics
            st.divider()
            st.subheader(f"Residual Diagnostics ({best_model})")
            test_actuals = ev["test_df"]["Sales"].values
            test_preds = preds_dict[best_model]
            diag = compute_residual_diagnostics(test_actuals, test_preds, ev["test_df"]["Date"].values)
            
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("Mean Residual Bias", f"{diag['mean_residual']:,.0f} units", help="Positive means model underpredicted on average")
            cd2.metric("Residual Std Dev", f"{diag['std_residual']:,.0f} units")
            cd3.metric("Durbin-Watson Stat", f"{diag['durbin_watson']}", help="~2.0 indicates no residual autocorrelation")
            
            fig_diag = plot_residual_diagnostics_charts(diag["residual_df"])
            st.plotly_chart(fig_diag, use_container_width=True)


# ==============================================================================
# PAGE 6: STRATEGIC BUSINESS INSIGHTS
# ==============================================================================
elif page == "6. Strategic Business Insights":
    st.title("💡 Strategic Automotive Business Insights")
    st.markdown("Evidence-based market findings, supply chain implications, and dealer inventory scheduling.")
    
    if require_clean_data():
        df = st.session_state.clean_df
        fc_df = None
        if st.session_state.latest_forecast_result is not None:
            fc_df = st.session_state.latest_forecast_result.to_dataframe()
            
        insights = generate_automotive_business_insights(
            df,
            forecast_df=fc_df,
            promotional_data_provided=(st.session_state.promotional_df is not None)
        )
        
        # Summary Metrics
        c_in1, c_in2, c_in3 = st.columns(3)
        with c_in1:
            cagr = insights.get("cagr_pct")
            st.metric("Historical CAGR", f"{cagr:+.1f}%" if cagr is not None else "N/A")
        with c_in2:
            st.metric("Top Demand Months", ", ".join(insights.get("top_months", [])))
        with c_in3:
            st.metric("Consolidation Months", ", ".join(insights.get("slack_months", [])))
            
        st.divider()
        
        # Actionable Insight Cards
        for item in insights.get("insights", []):
            with st.container():
                st.subheader(f"📌 {item.get('title')}")
                st.markdown(f"**Domain**: *{item.get('category')}* | Impact: `{item.get('impact_type')}`")
                st.info(item.get("detail"))
                st.write("")
                
        # Promotional Disclaimer
        st.divider()
        st.subheader("📢 Promotional Factor Assessment")
        st.warning(insights.get("promotional_statement"))


# ==============================================================================
# PAGE 7: EXPORT & EXECUTIVE REPORTS
# ==============================================================================
elif page == "7. Export & Executive Reports":
    st.title("📑 Export & Executive Reports")
    st.markdown("Generate and download executive intelligence summaries, PDF reports, and structured CSV datasets.")
    
    if require_clean_data():
        df = st.session_state.clean_df
        kpis = calculate_sales_kpis(df)
        
        fc_df = None
        if st.session_state.latest_forecast_result is not None:
            fc_df = st.session_state.latest_forecast_result.to_dataframe()
            
        eval_df = None
        if st.session_state.eval_results is not None:
            eval_df = st.session_state.eval_results.get("ranked_df")
            
        insights = generate_automotive_business_insights(df, forecast_df=fc_df)
        
        col_rep1, col_rep2 = st.columns(2)
        
        # Text / Markdown Report
        with col_rep1:
            st.subheader("1. Executive Summary Report (Text / Markdown)")
            text_rep = generate_text_executive_report(
                kpis=kpis,
                insights_dict=insights,
                eval_df=eval_df,
                forecast_df=fc_df,
                dataset_name=meta.get("dataset_name", "Indian Passenger Vehicle Sales")
            )
            st.text_area("Report Preview:", text_rep, height=350)
            
            st.download_button(
                label="📥 Download Text Report (.txt)",
                data=text_rep.encode("utf-8"),
                file_name=f"india_car_sales_executive_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

        # PDF Report
        with col_rep2:
            st.subheader("2. Formal PDF Intelligence Report")
            st.markdown("""
            Generates a publication-ready PDF document including:
            - Executive KPIs & Volume Summary
            - Held-out Model Evaluation Benchmark
            - 12-Month Future Forecast Table
            - Automotive Supply Chain Insights & Disclaimers
            """)
            
            if REPORTLAB_AVAILABLE:
                try:
                    pdf_bytes = generate_pdf_report_bytes(
                        kpis=kpis,
                        insights_dict=insights,
                        eval_df=eval_df,
                        forecast_df=fc_df,
                        dataset_name=meta.get("dataset_name", "Indian Passenger Vehicle Sales")
                    )
                    st.download_button(
                        label="📥 Download Executive PDF Report",
                        data=pdf_bytes,
                        file_name=f"india_car_sales_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated and ready for download!")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
            else:
                st.warning("ReportLab library is not available. Please install `reportlab` to enable PDF export.")

        # CSV Data Exports
        st.divider()
        st.subheader("3. Structured CSV Data Exports")
        
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            csv_clean = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned Sales CSV",
                data=csv_clean,
                file_name="cleaned_india_car_sales.csv",
                mime="text/csv"
            )
        with c_exp2:
            if fc_df is not None:
                csv_fc = fc_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Forecast Results CSV",
                    data=csv_fc,
                    file_name="forecast_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("Run forecast on Page 4 to export predictions.")
        with c_exp3:
            if eval_df is not None:
                csv_eval = eval_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Evaluation Metrics CSV",
                    data=csv_eval,
                    file_name="model_evaluation_metrics.csv",
                    mime="text/csv"
                )
            else:
                st.info("Run benchmark on Page 5 to export metrics.")
