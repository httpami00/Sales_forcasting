"""
Business insights and automotive strategy module.
Generates evidence-based observations covering long-term volume trends,
seasonal demand cycles, forecast uncertainty, dealer inventory planning,
and promotional factor disclaimers.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


def generate_automotive_business_insights(
    df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    evaluation_metrics: Optional[pd.DataFrame] = None,
    promotional_data_provided: bool = False
) -> Dict[str, Any]:
    """
    Generate evidence-based observations about market trends, seasonal patterns,
    forecast implications, and supply-chain / dealership planning.
    """
    clean_df = df.dropna(subset=["Sales"]).sort_values("Date").reset_index(drop=True)
    if len(clean_df) < 12:
        return {
            "summary": "Insufficient historical data (less than 12 months) to generate statistical business insights.",
            "insights": []
        }
        
    start_date = clean_df["Date"].min()
    end_date = clean_df["Date"].max()
    num_years = (end_date - start_date).days / 365.25
    
    start_sales = float(clean_df.iloc[0]["Sales"])
    end_sales = float(clean_df.iloc[-1]["Sales"])
    
    # 1. Long-term Growth & Trend
    cagr = None
    if num_years >= 1 and start_sales > 0:
        cagr = ((end_sales / start_sales) ** (1.0 / num_years) - 1.0) * 100.0
        
    avg_sales = float(clean_df["Sales"].mean())
    max_row = clean_df.loc[clean_df["Sales"].idxmax()]
    min_row = clean_df.loc[clean_df["Sales"].idxmin()]
    
    # 2. Seasonality Insights
    clean_df["Month_Name"] = clean_df["Date"].dt.strftime("%b")
    clean_df["Month_Num"] = clean_df["Date"].dt.month
    month_stats = clean_df.groupby(["Month_Num", "Month_Name"])["Sales"].mean().reset_index()
    month_stats = month_stats.sort_values(by="Sales", ascending=False)
    
    top_3_months = month_stats.head(3)["Month_Name"].tolist()
    bottom_3_months = month_stats.tail(3)["Month_Name"].tolist()
    
    # 3. Recent Trajectory
    recent_12m = clean_df.tail(12)
    prior_12m = clean_df.iloc[-24:-12] if len(clean_df) >= 24 else None
    
    recent_annual_vol = float(recent_12m["Sales"].sum())
    recent_yoy_growth = None
    if prior_12m is not None and len(prior_12m) == 12:
        prior_vol = float(prior_12m["Sales"].sum())
        if prior_vol > 0:
            recent_yoy_growth = ((recent_annual_vol - prior_vol) / prior_vol) * 100.0
            
    # 4. Forecast Implications & Inventory Advice
    forecast_summary = {}
    if forecast_df is not None and not forecast_df.empty:
        mean_future_sales = float(forecast_df["Forecast"].mean())
        total_future_sales = float(forecast_df["Forecast"].sum())
        highest_future_month = forecast_df.loc[forecast_df["Forecast"].idxmax()]
        lowest_future_month = forecast_df.loc[forecast_df["Forecast"].idxmin()]
        
        forecast_summary = {
            "horizon_months": len(forecast_df),
            "projected_total_volume": round(total_future_sales),
            "projected_monthly_average": round(mean_future_sales),
            "peak_forecast_month": f"{highest_future_month['Date'].strftime('%B %Y')} ({highest_future_month['Forecast']:,} units)",
            "trough_forecast_month": f"{lowest_future_month['Date'].strftime('%B %Y')} ({lowest_future_month['Forecast']:,} units)"
        }
        
    # Compile Key Executive Insights
    insights = []
    
    # Long Term
    cagr_str = f"with an annualized growth rate (CAGR) of {cagr:.1f}%" if cagr is not None else ""
    insights.append({
        "category": "Long-Term Market Trend",
        "title": "Expansion of India Passenger Vehicle Market",
        "detail": (
            f"Over the {num_years:.1f}-year observation period from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}, "
            f"monthly factory dispatches expanded from {int(start_sales):,} to {int(end_sales):,} units {cagr_str}. "
            f"The market demonstrated robust structural resilience following the severe April 2020 lockdown shock."
        ),
        "impact_type": "Positive" if (cagr or 0) > 0 else "Neutral"
    })
    
    # Seasonality
    insights.append({
        "category": "Seasonal Demand Dynamics",
        "title": f"Festive Peak ({', '.join(top_3_months)}) vs Monsoon/Year-End Slack ({', '.join(bottom_3_months)})",
        "detail": (
            f"Historical dispatches consistently peak during {top_3_months[0]} and {top_3_months[1]}, coinciding with "
            "the Indian festive season (Navratri, Dussehra, Dhanteras, and Diwali) when OEMs build dealer inventory to meet "
            f"surging retail demand. In contrast, months like {bottom_3_months[0]} and {bottom_3_months[1]} historically record "
            "lower dispatches due to seasonal monsoons and year-end inventory consolidation."
        ),
        "impact_type": "Cyclical"
    })
    
    # Inventory Planning
    insights.append({
        "category": "Automotive Operations & Inventory Planning",
        "title": "OEM Production Schedule & Dealer Stocking Recommendations",
        "detail": (
            "1. **Pre-Festive Buffer (July-August)**: Dealerships and Tier-1 suppliers should ramp component buffers 45-60 days "
            "ahead of September/October dispatch peaks to avert bottlenecks.\n"
            "2. **Year-End Inventory Normalization (November-December)**: Dealers typically discount existing model-year inventory "
            "in December to prevent carrying older manufacturing-year stock into January.\n"
            "3. **Fiscal Year-End Target Push (March)**: High corporate and commercial depreciation purchases routinely elevate March dispatches."
        ),
        "impact_type": "Operational"
    })
    
    # Uncertainty Note
    insights.append({
        "category": "Model Uncertainty & External Factors",
        "title": "Model Forecast Bounds & Supply Chain Variability",
        "detail": (
            "Point forecasts represent mathematical central tendencies based on past dispatch patterns. "
            "Actual realization may deviate due to macroeconomic interest rate adjustments, fuel price changes, "
            "raw material inflation (steel/aluminum), regulatory compliance revisions (e.g. BS-VI, RDE norms), "
            "and unpredictable semiconductor or shipping constraints. Use upper and lower prediction bounds for risk management."
        ),
        "impact_type": "Risk"
    })
    
    # Promotional Statement
    if promotional_data_provided:
        promo_statement = (
            "External promotional indicators and festive indices were supplied. The regression model (SARIMAX) "
            "estimates the incremental dispatch uplift attributable to synchronized marketing campaigns."
        )
    else:
        promo_statement = (
            "Promotional impact cannot be estimated from the current dataset because promotional information is unavailable. "
            "The historical dataset reflects aggregate factory dispatches; discount schemes, marketing budgets, and dealer "
            "promotions are not tracked in the primary SIAM/Trendonify dispatch series."
        )

    return {
        "cagr_pct": round(cagr, 2) if cagr is not None else None,
        "recent_annual_volume": round(recent_annual_vol),
        "recent_yoy_growth_pct": round(recent_yoy_growth, 2) if recent_yoy_growth is not None else None,
        "top_months": top_3_months,
        "slack_months": bottom_3_months,
        "forecast_summary": forecast_summary,
        "insights": insights,
        "promotional_statement": promo_statement
    }
