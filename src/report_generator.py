"""
Report generation module for India Car Sales Forecasting System.
Generates comprehensive executive summary text/markdown reports and
professional PDF reports using ReportLab.
"""

import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


def generate_text_executive_report(
    kpis: Dict[str, Any],
    insights_dict: Dict[str, Any],
    eval_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    dataset_name: str = "Indian Passenger Vehicle Sales Dataset"
) -> str:
    """
    Generate a formatted plain text / markdown executive intelligence report.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append("=" * 80)
    lines.append("INDIA CAR SALES FORECASTING & PREDICTIVE ANALYTICS EXECUTIVE REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated On: {now_str}")
    lines.append(f"Target Dataset: {dataset_name}")
    lines.append(f"Coverage: {kpis.get('start_month', 'N/A')} to {kpis.get('end_month', 'N/A')} ({kpis.get('total_months_count', 0)} monthly observations)")
    lines.append("-" * 80)
    lines.append("")
    
    # Section 1: Executive KPI Overview
    lines.append("1. EXECUTIVE SALES KPIs")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"- Cumulative Historical Sales:  {kpis.get('total_historical_sales', 0):,.0f} units")
    lines.append(f"- Average Monthly Sales:        {kpis.get('average_monthly_sales', 0):,.0f} units")
    lines.append(f"- Highest Recorded Month:       {kpis.get('highest_sales_month', 'N/A')} ({kpis.get('highest_sales_value', 0):,.0f} units)")
    lines.append(f"- Lowest Recorded Month:        {kpis.get('lowest_sales_month', 'N/A')} ({kpis.get('lowest_sales_value', 0):,.0f} units)")
    lines.append(f"- Latest Month Volume:          {kpis.get('latest_recorded_month', 'N/A')} ({kpis.get('latest_recorded_sales', 0):,.0f} units)")
    if kpis.get("latest_mom_change_pct") is not None:
        lines.append(f"- Latest Month-over-Month:      {kpis.get('latest_mom_change_pct'):+.2f}%")
    if kpis.get("latest_yoy_change_pct") is not None:
        lines.append(f"- Latest Year-over-Year:        {kpis.get('latest_yoy_change_pct'):+.2f}%")
    lines.append("")
    
    # Section 2: Model Benchmark & Evaluation
    if eval_df is not None and not eval_df.empty:
        lines.append("2. OUT-OF-SAMPLE MODEL BENCHMARK (CHRONOLOGICAL TEST PERIOD)")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"{'Rank':<6}{'Model':<30}{'MAE':<12}{'RMSE':<12}{'WAPE (%)':<12}{'MAPE (%)':<10}")
        lines.append("-" * 80)
        for _, row in eval_df.iterrows():
            wape_str = f"{row['WAPE_Pct']:.2f}%" if pd.notnull(row.get('WAPE_Pct')) else "N/A"
            mape_str = f"{row['MAPE_Pct']:.2f}%" if pd.notnull(row.get('MAPE_Pct')) else "N/A"
            lines.append(
                f"{row.get('Rank', '-'):<6}{str(row.get('Model', ''))[:28]:<30}"
                f"{row.get('MAE', 0):<12,.0f}{row.get('RMSE', 0):<12,.0f}{wape_str:<12}{mape_str:<10}"
            )
        lines.append("")
        
    # Section 3: Future Demand Projections
    if forecast_df is not None and not forecast_df.empty:
        lines.append("3. FUTURE DEMAND FORECAST PROJECTIONS")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"Model Employed: {forecast_df.iloc[0].get('Model', 'Selected Model')}")
        lines.append(f"{'Month':<12}{'Forecast (Units)':<18}{'95% Lower Bound':<18}{'95% Upper Bound':<18}{'MoM Change':<12}")
        lines.append("-" * 80)
        for _, row in forecast_df.iterrows():
            m_str = row["Date"].strftime("%b %Y") if hasattr(row["Date"], "strftime") else str(row["Date"])[:7]
            fc_str = f"{row['Forecast']:,.0f}"
            lo_str = f"{row['Lower_Bound']:,.0f}" if pd.notnull(row.get("Lower_Bound")) else "N/A"
            up_str = f"{row['Upper_Bound']:,.0f}" if pd.notnull(row.get("Upper_Bound")) else "N/A"
            mom_str = f"{row['MoM_Change_Pct']:+.2f}%" if pd.notnull(row.get("MoM_Change_Pct")) else "N/A"
            lines.append(f"{m_str:<12}{fc_str:<18}{lo_str:<18}{up_str:<18}{mom_str:<12}")
        lines.append("")
        
    # Section 4: Automotive Business Insights
    lines.append("4. STRATEGIC AUTOMOTIVE BUSINESS INSIGHTS")
    lines.append("--------------------------------------------------------------------------------")
    for ins in insights_dict.get("insights", []):
        lines.append(f"[{ins.get('category', 'Insight').upper()}]: {ins.get('title', '')}")
        lines.append(f"{ins.get('detail', '')}")
        lines.append("")
        
    lines.append("5. PROMOTIONAL FACTOR STATEMENT")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(insights_dict.get("promotional_statement", "N/A"))
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT - India Car Sales Forecasting & Predictive Analytics System")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_pdf_report_bytes(
    kpis: Dict[str, Any],
    insights_dict: Dict[str, Any],
    eval_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    dataset_name: str = "Indian Passenger Vehicle Sales Dataset"
) -> bytes:
    """
    Generate a polished PDF executive report in bytes using ReportLab.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not available in the current environment.")
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A")
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )
    
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155")
    )
    
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )
    
    story = []
    
    # Document Header
    story.append(Paragraph("India Car Sales Forecasting & Predictive Analytics", title_style))
    story.append(Spacer(1, 4))
    date_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    story.append(Paragraph(f"Executive Intelligence Report | Dataset: {dataset_name} | Generated: {date_str}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=12))
    
    # 1. KPI Overview Table
    story.append(Paragraph("1. Executive Sales KPIs", heading_style))
    kpi_data = [
        [
            Paragraph("<b>Total Sales:</b>", table_cell_bold),
            Paragraph(f"{kpis.get('total_historical_sales', 0):,.0f} units", table_cell_style),
            Paragraph("<b>Avg Monthly:</b>", table_cell_bold),
            Paragraph(f"{kpis.get('average_monthly_sales', 0):,.0f} units", table_cell_style),
        ],
        [
            Paragraph("<b>Highest Month:</b>", table_cell_bold),
            Paragraph(f"{kpis.get('highest_sales_month', 'N/A')} ({kpis.get('highest_sales_value', 0):,.0f})", table_cell_style),
            Paragraph("<b>Lowest Month:</b>", table_cell_bold),
            Paragraph(f"{kpis.get('lowest_sales_month', 'N/A')} ({kpis.get('lowest_sales_value', 0):,.0f})", table_cell_style),
        ],
        [
            Paragraph("<b>Latest Month:</b>", table_cell_bold),
            Paragraph(f"{kpis.get('latest_recorded_month', 'N/A')} ({kpis.get('latest_recorded_sales', 0):,.0f})", table_cell_style),
            Paragraph("<b>Latest Growth:</b>", table_cell_bold),
            Paragraph(f"MoM: {kpis.get('latest_mom_change_pct', 0):+.1f}% | YoY: {kpis.get('latest_yoy_change_pct', 0):+.1f}%", table_cell_style),
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[1.3 * inch, 2.2 * inch, 1.3 * inch, 2.2 * inch])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))
    
    # 2. Model Evaluation Table
    if eval_df is not None and not eval_df.empty:
        story.append(Paragraph("2. Model Evaluation Benchmark (Held-out Test Period)", heading_style))
        eval_headers = ["Rank", "Model", "MAE (Units)", "RMSE", "WAPE (%)", "MAPE (%)"]
        eval_rows = [[Paragraph(f"<b>{h}</b>", table_cell_bold) for h in eval_headers]]
        for _, r in eval_df.iterrows():
            wape_val = f"{r['WAPE_Pct']:.2f}%" if pd.notnull(r.get('WAPE_Pct')) else "N/A"
            mape_val = f"{r['MAPE_Pct']:.2f}%" if pd.notnull(r.get('MAPE_Pct')) else "N/A"
            eval_rows.append([
                Paragraph(str(r.get('Rank', '-')), table_cell_style),
                Paragraph(str(r.get('Model', ''))[:24], table_cell_style),
                Paragraph(f"{r.get('MAE', 0):,.0f}", table_cell_style),
                Paragraph(f"{r.get('RMSE', 0):,.0f}", table_cell_style),
                Paragraph(wape_val, table_cell_style),
                Paragraph(mape_val, table_cell_style),
            ])
        t_eval = Table(eval_rows, colWidths=[0.6 * inch, 2.4 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch, 0.9 * inch])
        t_eval.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFF6FF")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_eval)
        story.append(Spacer(1, 10))
        
    # 3. Future Forecast Projections
    if forecast_df is not None and not forecast_df.empty:
        story.append(Paragraph("3. Future Demand Forecast", heading_style))
        fc_headers = ["Month", "Forecast", "95% Lower Bound", "95% Upper Bound", "MoM (%)"]
        fc_rows = [[Paragraph(f"<b>{h}</b>", table_cell_bold) for h in fc_headers]]
        for _, r in forecast_df.head(12).iterrows():
            m_s = r["Date"].strftime("%b %Y") if hasattr(r["Date"], "strftime") else str(r["Date"])[:7]
            lo_s = f"{r['Lower_Bound']:,.0f}" if pd.notnull(r.get("Lower_Bound")) else "N/A"
            up_s = f"{r['Upper_Bound']:,.0f}" if pd.notnull(r.get("Upper_Bound")) else "N/A"
            mom_s = f"{r['MoM_Change_Pct']:+.1f}%" if pd.notnull(r.get("MoM_Change_Pct")) else "N/A"
            fc_rows.append([
                Paragraph(m_s, table_cell_style),
                Paragraph(f"{r['Forecast']:,.0f}", table_cell_style),
                Paragraph(lo_s, table_cell_style),
                Paragraph(up_s, table_cell_style),
                Paragraph(mom_s, table_cell_style),
            ])
        t_fc = Table(fc_rows, colWidths=[1.2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch, 1.3 * inch])
        t_fc.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECFDF5")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#A7F3D0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_fc)
        story.append(Spacer(1, 10))
        
    # 4. Strategic Business Insights
    story.append(Paragraph("4. Strategic Business Insights & Planning Recommendations", heading_style))
    for ins in insights_dict.get("insights", [])[:4]:
        story.append(Paragraph(f"<b>{ins.get('category', 'Insight')}: {ins.get('title', '')}</b>", table_cell_bold))
        story.append(Paragraph(ins.get("detail", ""), body_style))
        story.append(Spacer(1, 4))
        
    # Promotional Disclaimer
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Promotional Impact Note:</b>", table_cell_bold))
    story.append(Paragraph(insights_dict.get("promotional_statement", "N/A"), body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
