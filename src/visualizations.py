"""
Interactive Plotly visualization module for India Car Sales.
Creates clean, interactive, responsive charts with tooltips, clear axis labels,
and distinct visual styles for actuals, test predictions, and future forecast intervals.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Elegant unified color palette
COLOR_PRIMARY = "#2563EB"      # Deep Royal Blue (Historical Actuals)
COLOR_TEST_ACTUAL = "#0F172A"  # Dark Slate (Test Actuals)
COLOR_FORECAST = "#10B981"     # Emerald Green (Future Forecast)
COLOR_INTERVAL = "rgba(16, 185, 129, 0.18)" # Soft Emerald Tint
COLOR_ANOMALY = "#EF4444"      # Crimson Red (Anomalies)
COLOR_TREND = "#F59E0B"        # Amber / Orange (Rolling Trend)
COLOR_MUTED = "#94A3B8"        # Slate Gray (Context / Background)

MODEL_PALETTE = {
    "Naive (Last Value)": "#6B7280",
    "Seasonal Naive (t-12)": "#8B5CF6",
    "ARIMA": "#F59E0B",
    "SARIMA": "#3B82F6",
    "Prophet": "#EC4899"
}


def _apply_clean_layout(fig: go.Figure, title: str, x_title: str = "Date", y_title: str = "Sales (Units)") -> go.Figure:
    """Apply consistent styling, typography, and gridlines to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 18, "family": "Inter, Roboto, sans-serif"}
        },
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=70, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickformat=","
        )
    )
    return fig


def plot_historical_sales_and_rolling(
    df: pd.DataFrame,
    show_rolling_3m: bool = True,
    show_rolling_6m: bool = False,
    show_rolling_12m: bool = True
) -> go.Figure:
    """
    Plot full historical monthly sales alongside optional rolling moving averages.
    """
    fig = go.Figure()
    
    # Actual Sales Line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Sales"],
        mode="lines+markers",
        name="Actual Monthly Dispatches",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        marker=dict(size=4, color=COLOR_PRIMARY),
        hovertemplate="<b>Date</b>: %{x|%b %Y}<br><b>Sales</b>: %{y:,.0f} units<extra></extra>"
    ))
    
    # Rolling Moving Averages
    if show_rolling_3m and "Rolling_3M" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Rolling_3M"],
            mode="lines",
            name="3-Month Rolling Average",
            line=dict(color="#06B6D4", width=2, dash="dash"),
            hovertemplate="3M Avg: %{y:,.0f}<extra></extra>"
        ))
    if show_rolling_6m and "Rolling_6M" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Rolling_6M"],
            mode="lines",
            name="6-Month Rolling Average",
            line=dict(color="#8B5CF6", width=2, dash="dot"),
            hovertemplate="6M Avg: %{y:,.0f}<extra></extra>"
        ))
    if show_rolling_12m and "Rolling_12M" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Rolling_12M"],
            mode="lines",
            name="12-Month Rolling Trend",
            line=dict(color=COLOR_TREND, width=2.5),
            hovertemplate="12M Trend: %{y:,.0f}<extra></extra>"
        ))
        
    # Highlight Peak and Lockdown Trough with Annotations
    if not df.empty and df["Sales"].dropna().count() > 0:
        max_idx = df["Sales"].idxmax()
        min_idx = df["Sales"].idxmin()
        max_row = df.loc[max_idx]
        min_row = df.loc[min_idx]
        
        fig.add_annotation(
            x=max_row["Date"],
            y=max_row["Sales"],
            text=f"Record Peak: {int(max_row['Sales']):,}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#059669",
            ax=0,
            ay=-40,
            bgcolor="#ECFDF5",
            bordercolor="#10B981",
            font=dict(size=11, color="#065F46")
        )
        
        if min_row["Sales"] == 0 or min_row["Sales"] < 50000:
            fig.add_annotation(
                x=min_row["Date"],
                y=min_row["Sales"],
                text=f"National Lockdown ({min_row['Date'].strftime('%b %Y')}): {int(min_row['Sales']):,}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=COLOR_ANOMALY,
                ax=0,
                ay=40,
                bgcolor="#FEF2F2",
                bordercolor=COLOR_ANOMALY,
                font=dict(size=11, color="#991B1B")
            )
            
    return _apply_clean_layout(fig, "India Monthly Passenger Vehicle Dispatches & Rolling Trends")


def plot_anomaly_detection_scatter(df_anomalies: pd.DataFrame) -> go.Figure:
    """
    Plot sales trend with statistical anomalies highlighted via distinctive markers.
    """
    fig = go.Figure()
    
    # Normal trend line
    fig.add_trace(go.Scatter(
        x=df_anomalies["Date"],
        y=df_anomalies["Sales"],
        mode="lines",
        name="Historical Monthly Sales",
        line=dict(color=COLOR_PRIMARY, width=2),
        hoverinfo="skip"
    ))
    
    # Regular points
    normal_mask = ~df_anomalies["Is_Anomaly"]
    fig.add_trace(go.Scatter(
        x=df_anomalies.loc[normal_mask, "Date"],
        y=df_anomalies.loc[normal_mask, "Sales"],
        mode="markers",
        name="Regular Observations",
        marker=dict(size=6, color=COLOR_PRIMARY, opacity=0.7),
        hovertemplate="<b>%{x|%b %Y}</b><br>Sales: %{y:,.0f} units<br>Status: Normal<extra></extra>"
    ))
    
    # Anomalous points
    anomaly_mask = df_anomalies["Is_Anomaly"]
    if anomaly_mask.any():
        anom_rows = df_anomalies[anomaly_mask]
        fig.add_trace(go.Scatter(
            x=anom_rows["Date"],
            y=anom_rows["Sales"],
            mode="markers",
            name="Statistical Anomalies",
            marker=dict(
                size=12,
                color=COLOR_ANOMALY,
                symbol="diamond-wide",
                line=dict(color="#7F1D1D", width=2)
            ),
            hovertemplate=(
                "<b>%{x|%b %Y} - ANOMALY DETECTED</b><br>"
                "Sales: %{y:,.0f} units<br>"
                "Direction: %{customdata[0]}<br>"
                "Detail: %{customdata[1]}<extra></extra>"
            ),
            customdata=np.stack((anom_rows["Anomaly_Direction"], anom_rows["Anomaly_Reason"]), axis=-1)
        ))
        
    return _apply_clean_layout(fig, "Automated Statistical Anomaly Detection (IQR / Z-Score)")


def plot_seasonality_month_bars(monthly_agg: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing average monthly car sales by calendar month (Jan-Dec).
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_agg["Month_Name"],
        y=monthly_agg["Mean"],
        name="Average Monthly Sales",
        marker=dict(
            color=monthly_agg["Mean"],
            colorscale="Blues",
            showscale=False
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=monthly_agg["Max"] - monthly_agg["Mean"],
            arrayminus=monthly_agg["Mean"] - monthly_agg["Min"],
            color="#64748B",
            thickness=1.5,
            width=5
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Average Dispatches: %{y:,.0f} units<br>"
            "Historical Range: Min to Max (bars)<extra></extra>"
        )
    ))
    
    return _apply_clean_layout(fig, "Average Passenger Vehicle Sales by Calendar Month (with Range)", "Calendar Month")


def plot_seasonality_heatmap(pivot_heatmap: pd.DataFrame) -> go.Figure:
    """
    Heatmap comparing Years against Months.
    """
    fig = go.Figure(data=go.Heatmap(
        z=pivot_heatmap.values,
        x=list(pivot_heatmap.columns),
        y=list(pivot_heatmap.index),
        colorscale="Blues",
        colorbar=dict(title="Units Sold", tickformat=","),
        hovertemplate="<b>Year</b>: %{y}<br><b>Month</b>: %{x}<br><b>Sales</b>: %{z:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title={"text": "<b>Yearly vs Monthly Dispatch Heatmap Matrix</b>", "font": {"size": 16}},
        template="plotly_white",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year", dtick=1),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    return fig


def plot_seasonal_decomposition_components(decomp: Any) -> go.Figure:
    """
    4-panel interactive decomposition subplot (Observed, Trend, Seasonal, Residual).
    """
    dates = decomp.observed.index
    
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "<b>Observed Dispatches</b>",
            "<b>Underlying Trend Component</b>",
            "<b>Repeating Seasonal Component (12-Month Cycle)</b>",
            "<b>Residual (Irregular / Random Component)</b>"
        ]
    )
    
    # 1. Observed
    fig.add_trace(go.Scatter(x=dates, y=decomp.observed, mode="lines", name="Observed", line=dict(color=COLOR_PRIMARY, width=2)), row=1, col=1)
    # 2. Trend
    fig.add_trace(go.Scatter(x=dates, y=decomp.trend, mode="lines", name="Trend", line=dict(color=COLOR_TREND, width=2.5)), row=2, col=1)
    # 3. Seasonal
    fig.add_trace(go.Scatter(x=dates, y=decomp.seasonal, mode="lines", name="Seasonal", line=dict(color="#10B981", width=2)), row=3, col=1)
    # 4. Residual
    fig.add_trace(go.Scatter(x=dates, y=decomp.resid, mode="markers", name="Residual", marker=dict(size=4, color="#64748B")), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", row=4, col=1)
    
    fig.update_layout(
        height=700,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=50)
    )
    fig.update_yaxes(tickformat=",")
    return fig


def plot_test_evaluation_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    show_recent_train_months: int = 18
) -> go.Figure:
    """
    Plot actual held-out test data against out-of-sample predictions from all models.
    """
    fig = go.Figure()
    
    # Recent training history context
    train_slice = train_df.tail(show_recent_train_months)
    fig.add_trace(go.Scatter(
        x=train_slice["Date"],
        y=train_slice["Sales"],
        mode="lines+markers",
        name="Training History",
        line=dict(color=COLOR_MUTED, width=1.5, dash="dot"),
        marker=dict(size=4, color=COLOR_MUTED),
        hovertemplate="Train Actual: %{y:,.0f}<extra></extra>"
    ))
    
    # Ground Truth Test Actuals
    fig.add_trace(go.Scatter(
        x=test_df["Date"],
        y=test_df["Sales"],
        mode="lines+markers",
        name="Test Ground Truth (Actual)",
        line=dict(color=COLOR_TEST_ACTUAL, width=3.5),
        marker=dict(size=7, color=COLOR_TEST_ACTUAL),
        hovertemplate="<b>Actual Sales</b>: %{y:,.0f} units<extra></extra>"
    ))
    
    # Model Predictions
    colors = ["#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899", "#10B981"]
    dash_styles = ["dash", "dashdot", "longdash", "dot", "dash"]
    
    for i, (m_name, y_preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        style = dash_styles[i % len(dash_styles)]
        
        fig.add_trace(go.Scatter(
            x=test_df["Date"],
            y=y_preds,
            mode="lines+markers",
            name=f"Predicted: {m_name}",
            line=dict(color=color, width=2, dash=style),
            marker=dict(size=5, symbol="circle-open"),
            hovertemplate=f"<b>{m_name}</b>: %{{y:,.0f}}<extra></extra>"
        ))
        
    return _apply_clean_layout(fig, "Held-Out Test Period: Actual vs Predicted Model Benchmark")


def plot_future_forecast_chart(
    historical_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    forecast_df: pd.DataFrame,
    model_name: str,
    show_historical_months: int = 24
) -> go.Figure:
    """
    Interactive Plotly forecast chart showing:
    - Historical actual observations
    - Test-period actuals / test predictions (if available)
    - Future monthly forecasts
    - Upper and Lower prediction intervals (when available)
    With distinct visual styling and clear labels as model estimates.
    """
    fig = go.Figure()
    
    # Historical observations
    hist_slice = historical_df.tail(show_historical_months)
    fig.add_trace(go.Scatter(
        x=hist_slice["Date"],
        y=hist_slice["Sales"],
        mode="lines+markers",
        name="Historical Actual Dispatches",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        marker=dict(size=4, color=COLOR_PRIMARY),
        hovertemplate="<b>Actual</b>: %{y:,.0f} units<br>Date: %{x|%b %Y}<extra></extra>"
    ))
    
    # Prediction Interval (Upper and Lower bounds shaded)
    has_bounds = "Lower_Bound" in forecast_df.columns and "Upper_Bound" in forecast_df.columns and forecast_df["Lower_Bound"].notna().any()
    
    if has_bounds:
        valid_b = forecast_df.dropna(subset=["Lower_Bound", "Upper_Bound"])
        # Upper bound line
        fig.add_trace(go.Scatter(
            x=valid_b["Date"],
            y=valid_b["Upper_Bound"],
            mode="lines",
            line=dict(width=0),
            name="95% Upper Prediction Bound",
            showlegend=False,
            hoverinfo="skip"
        ))
        # Lower bound line filled to upper
        fig.add_trace(go.Scatter(
            x=valid_b["Date"],
            y=valid_b["Lower_Bound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COLOR_INTERVAL,
            name="95% Prediction Confidence Interval",
            hoverinfo="skip"
        ))
        
    # Future Forecast Line
    # Prepend the last historical point so the forecast line connects continuously
    last_hist_row = hist_slice.iloc[-1]
    connect_dates = pd.concat([pd.Series([last_hist_row["Date"]]), forecast_df["Date"]]).reset_index(drop=True)
    connect_values = pd.concat([pd.Series([last_hist_row["Sales"]]), forecast_df["Forecast"]]).reset_index(drop=True)
    
    fig.add_trace(go.Scatter(
        x=connect_dates,
        y=connect_values,
        mode="lines+markers",
        name=f"Future Forecast: {model_name} (Model Estimate)",
        line=dict(color=COLOR_FORECAST, width=3, dash="dash"),
        marker=dict(size=6, color=COLOR_FORECAST, symbol="diamond"),
        hovertemplate="<b>Projected Forecast (Estimate)</b>: %{y:,.0f} units<br>Month: %{x|%b %Y}<extra></extra>"
    ))
    
    # Vertical divider annotation separating history from future projection
    split_date = last_hist_row["Date"]
    fig.add_vline(
        x=split_date,
        line_width=1.5,
        line_dash="dot",
        line_color="#94A3B8"
    )
    
    fig.add_annotation(
        x=split_date,
        y=float(forecast_df["Forecast"].max()),
        text="<b>Forecast Horizon Begins ➔</b>",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font=dict(size=11, color="#059669")
    )
    
    return _apply_clean_layout(fig, f"India Car Sales Demand Forecast ({model_name}) with Prediction Intervals")


def plot_residual_diagnostics_charts(res_df: pd.DataFrame) -> go.Figure:
    """
    Subplots for residual time series and residual error distribution histogram.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["<b>Residuals Across Time (Actual - Predicted)</b>", "<b>Residual Error Distribution</b>"],
        column_widths=[0.6, 0.4]
    )
    
    # Time Series of Residuals
    x_axis = res_df["Date"] if "Date" in res_df.columns else res_df.index
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=res_df["Residual"],
        mode="lines+markers",
        name="Residuals",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=5)
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", row=1, col=1)
    
    # Histogram of Residuals
    fig.add_trace(go.Histogram(
        x=res_df["Residual"],
        nbinsx=15,
        name="Distribution",
        marker=dict(color="#8B5CF6", line=dict(color="#4C1D95", width=1))
    ), row=1, col=2)
    
    fig.update_layout(
        template="plotly_white",
        height=380,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=40)
    )
    return fig
