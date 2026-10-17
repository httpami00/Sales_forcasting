# Sample and Companion Datasets

This directory contains reference datasets designed for testing, demonstration, and exogenous factor analysis:

1. **`synthetic_car_sales_demo.csv`**:
   - Explicitly marked synthetic data (`Data_Type = SYNTHETIC_DEMO`).
   - Alternative column names (`Month`, `Vehicle_Sales`) to demonstrate column mapping and validation.
   - Used only as a fallback or for quick mock runs.
   - **Never mixed or blended with real historical observations.**

2. **`sample_promotional_data.csv`**:
   - Monthly external features:
     - `Promo_Index`: Normalized marketing and dealer incentive index (scale 0–10).
     - `Festive_Season`: Binary flag (1 for August–November festive season encompassing Ganesh Chaturthi, Onam, Navratri, Dussehra, Diwali, Dhanteras).
     - `Year_End_Discount`: Binary flag (1 for December model-year clearance discounts).
   - Includes future projection period (2025) to allow SARIMAX exog forecasting.
