# Indian Automotive Industry Sales Dataset

## Source & Attribution
- **Primary Source**: Society of Indian Automobile Manufacturers (SIAM) / Industry Monthly Dispatches Portal
- **Referenced By**: [Trendonify India Car Sales](https://trendonify.com/india/car-sales), AutoPunditz, FADA Research
- **Domain**: Automotive Industry / Passenger Vehicle Market in India
- **Observation Frequency**: Monthly (`MS` - Month Start)
- **Time Range**: January 2018 (`2018-01-01`) through December 2024 (`2024-12-01`)
- **Total Historical Observations**: 84 months

## Critical Metric Definition
> **Important Distinction**:
> The primary metric recorded in this dataset represents **Passenger Vehicle Domestic Factory Dispatches (Wholesale Sales)** from automotive manufacturers (OEMs) to authorized dealerships across India.
> 
> - **Wholesale Dispatches (SIAM metric)**: Vehicles shipped from manufacturing plants to dealership inventory. This is the official metric tracked by SIAM and historical forecasting benchmarks.
> - **Retail Registrations (FADA / Vahan metric)**: Vehicles registered at regional transport offices (RTOs) upon end-consumer delivery.
> - **Zero-Sales Period Note**: The `2020-04-01` observation of `0` is an authentic real-world event reflecting the strict nationwide COVID-19 lockdown in India when all automotive production and dealer dispatches were completely suspended.

## Column Dictionary
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Date` | `YYYY-MM-DD` | First day of the observation month |
| `Sales` | `Integer` | Total domestic passenger vehicles dispatched (units) |

## Dataset Limitations
1. Luxury automakers (such as BMW, Mercedes-Benz, JLR, Volvo) do not report dispatches monthly to SIAM and are excluded from standard industry monthly totals.
2. Figures represent nationwide aggregate dispatches and do not reflect regional/state-level variations.
3. Macro shocks (such as the April 2020 lockdown, May 2021 COVID wave 2, and September 2021 global semiconductor shortage) represent structural disruptions rather than routine seasonal fluctuations.
