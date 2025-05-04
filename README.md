## CO₂ Forecast Hybrid

*A Machine Learning and Time Series Approach to Ground-Level CO₂ Forecasting*

### 🌍 Overview

**CO₂ Forecast Hybrid** is a predictive modeling project that estimates future monthly average ground-level CO₂ concentrations (in ppm) using a **hybrid approach** combining machine learning and time series analysis. This project is motivated by the increasing need to forecast atmospheric CO₂ for environmental planning and climate policy support.

The model utilizes:
- **Random Forest** (Supervised Machine Learning)
- **SARIMA** (Seasonal Autoregressive Integrated Moving Average, a statistical time series model)

These methods are integrated to leverage both the non-linear relationships in environmental data and the temporal dependencies in CO₂ trends.  
### 🎯 Objective

To build an intelligent system that can:
- Learn patterns from historical CO₂, meteorological, and atmospheric data
- Forecast future CO₂ concentrations with improved accuracy
- Adapt to non-linear, multivariate, and seasonally varying signals


### 🧠 Modeling Approach

| Component       | Method        | Field             |
|----------------|---------------|-------------------|
| Random Forest  | Supervised ML | Machine Learning  |
| SARIMA         | Parametric    | Time Series Stats |
| Combined Model | RF + SARIMA   | Hybrid Approach   |


### 🔁 Workflow

```text
Start
└──> Input Parameters
└──> Random Forest Model
└──> RF Output Files
└──> SARIMA Forecasting
└──> Output Forecast
└──> Optional: Seasonal Decomposition
End
```


### 📥 Input Parameters

| Parameter   | Description                          | Role    | Source   |
|-------------|--------------------------------------|---------|----------|
| `co2obs`    | Ground-level CO₂ concentration       | Target  | BMKG     |
| `tcco2`     | Total column CO₂ (satellite)         | Feature | NASA     |
| `tcco_1e4`  | Total column CO (×10⁴)               | Feature | ECMWF    |
| `tcch4_1e4` | Total column CH₄ (×10⁴)              | Feature | ECMWF    |
| `u10`       | Zonal wind at 10m                    | Feature | ECMWF    |
| `v10`       | Meridional wind at 10m               | Feature | ECMWF    |
| `t2m`       | Temperature at 2m                    | Feature | ECMWF    |
| `mslp`      | Mean sea level pressure              | Feature | ECMWF    |



### 📊 Why a Hybrid Model?

| Limitation of SARIMA                          | How RF Helps                                   |
|-----------------------------------------------|-------------------------------------------------|
| Assumes linear relationships                  | Learns non-linear interactions                 |
| Cannot handle many predictors                 | Uses high-dimensional input features           |
| Poor with abrupt changes                      | Captures external forcing (e.g., temperature)  |
| No spatial/multivariate integration           | Ingests diverse datasets                       |

| Limitation of Random Forest                   | How SARIMA Helps                               |
|-----------------------------------------------|-------------------------------------------------|
| Lacks time structure modeling                 | Adds temporal dynamics and trends              |
| No forecasting horizon                        | Projects into the future                       |
| Ignores autocorrelation                       | Models serial dependence in time               |



### 🔍 Feature Importance and Contribution

Each feature contributes uniquely to the CO₂ variability at the Bukit Kototabang (BKT) site:

- **tcco2**: Indicates regional CO₂ burdens, useful for surface inference
- **tcch4_1e4**: Helps track sources from wetlands and fires
- **v10 / u10**: Capture monsoonal and synoptic transport influences
- **t2m**: Affects vertical mixing and diurnal variability
- **mslp**: Encodes atmospheric stability and seasonal pressure patterns
- **tcco_1e4**: Traces combustion-related events (e.g., biomass burning)



### 📁 Repository Contents


```text
CO2-Forecast-Hybrid/
├── data/        → Raw and processed input datasets
├── notebooks/   → Jupyter notebooks for training, analysis, and plotting
├── scripts/     → Main scripts for RF and SARIMA execution
└── README.md    → Project overview
```



### 📌 Notes

- All models are trained using open-source data.
- The model is designed to be extended with additional atmospheric predictors.
- Optional seasonal decomposition is available to analyze periodic patterns.




### 📬 Contact

For feedback, questions, or collaboration inquiries, please contact:  
📧 [alberth.nahas@bmkg.go.id]



### 📢 Citation

If you use this model or repository, please cite:

> “Hybrid Model for CO₂ Forecast” by [Alberth Nahas], 2025.  
> A hybrid ML and time series framework for forecasting atmospheric CO₂.

