# SolarVision — Time Series Module

Intelligent monitoring support for solar PV plants: **24–48h power forecasting**, **anomaly alerts**, and a **transparent cleaning cost–benefit decision**.

> Part of the SolarVision project (Feature 1). Thermal fault detection (Feature 2) is a separate module.

---

## Overview

SolarVision turns historical PV production and environmental data into operational signals:

| Capability | Description |
|------------|-------------|
| **Forecasting** | 24h-ahead (and optional 48h) PV power prediction |
| **Anomaly detection** | Sustained under-production vs forecast (daytime rules) |
| **Cleaning decision** | Scenario-based cost–benefit using user price/cost inputs |
| **API + Dashboard** | FastAPI endpoints + Streamlit UI |

**Data source:** [Hourly PV Power and Dust Accumulation Dataset — Laghouat, Algeria (2022)](https://data.mendeley.com/datasets/4jdt83yy43) only. No external APIs or invented sensors.

---

## Project structure

```text
├── 01_preprocessing.ipynb      # Features, targets, temporal split
├── 02_training.ipynb           # 24h models, MLflow, Optuna, SHAP
├── 02b_training_48h.ipynb      # Light 48h-ahead training
├── 03_anomaly_cleaning.ipynb   # Alerts + cleaning economics
├── api.py                      # FastAPI service
├── app_streamlit.py            # Streamlit dashboard
├── Laghouat_2022_final.csv     # Dataset (download separately if needed)
├── processed_data/             # Generated parquet + meta
├── models/                     # Saved joblib models
└── figures/                    # SHAP, anomaly plots
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn lightgbm optuna mlflow shap torch
pip install pyarrow matplotlib joblib fastapi uvicorn streamlit pydantic
```

Python 3.10+ recommended.

---

## Quick start

### 1. Data

Place `Laghouat_2022_final.csv` in the project root  
(download from the Mendeley dataset linked above).

### 2. Notebooks (in order)

```text
01_preprocessing.ipynb
        ↓
02_training.ipynb
        ↓
02b_training_48h.ipynb    # optional
        ↓
03_anomaly_cleaning.ipynb
```

### 3. API

```bash
uvicorn api:app --reload --port 8000
```

- Health: http://localhost:8000/health  
- Interactive docs: http://localhost:8000/docs  

| Endpoint | Method | Role |
|----------|--------|------|
| `/health` | GET | Model load status |
| `/forecast` | POST | Power prediction from feature map |
| `/anomaly` | POST | Residual-based sustained alerts |
| `/cleaning` | POST | Cost–benefit Clean / Wait |

### 4. Dashboard

```bash
streamlit run app_streamlit.py
```

Shows test forecast vs actual, anomaly summary, and cleaning decision controls.

---

## Modeling approach

### Features (from dataset only)

- Calendar & cyclic encodings (hour, day-of-year)
- Power lags & rolling statistics (1h–72h)
- Dust density (PCHIP-reconstructed) + lags/rolling
- Weather fields present in the CSV (temperature, RH, wind, clouds, precipitation, irradiance when available)
- Daytime flag (`G_POA > 5` when irradiance exists)

### Targets

- **24h-ahead:** `P_dusty` shifted −24 hours (main operational target)
- **48h-ahead:** same pipeline, longer horizon (`02b`)
- Nowcast-style comparison to published MATLAB work is **reference only** (different task)

### Model families compared

| Family | Models |
|--------|--------|
| Linear baseline | Ridge |
| Tree ensembles | Random Forest, LightGBM (+ Optuna) |
| Deep learning | LSTM (lags + static features) |
| Baseline | Persistence (lag 24h / 48h) |

- Experiments tracked with **MLflow**
- Hyperparameters: **Optuna** + time-series CV on LightGBM
- Metrics: **MAE, RMSE, MAPE (daytime), R²**
- Explainability: **SHAP**
- Chronological **70 / 15 / 15** split (no shuffle)

Persistence is always reported: PV series are strongly daily-periodic.

---

## Anomaly & cleaning logic

**Anomaly (default):** daytime hour where actual power is ≥20% below forecast, and this holds for ≥3 of the last 6 hours.

Interpretation: *unexpected drop vs model expectation* — **not** a verified root cause (dust, cloud, shadow, etc.).

**Cleaning decision:**

```text
net_benefit = (energy_loss_kwh × recoverable_fraction × electricity_price) − cleaning_cost
decision    = Clean if net_benefit > 0 else Wait
```

Energy totals can be taken from the test window; price, cost, recovery fraction, and plant scale are **user inputs**.

---

## Reference benchmark (MATLAB / Seghir et al.)

Published pipeline on the same site (nowcast-style evaluation, daytime N≈341):

| Model | RMSE (W) | MAE (W) | R² |
|-------|----------|---------|-----|
| Random Forest (best single) | ~4.46 | ~2.71 | ~0.96 |
| Hybrid ensemble | ~4.93 | ~3.23 | ~0.95 |

Our primary task is **multi-hour-ahead forecasting**, so these numbers are a methodological reference, not a required match.

---

## Limitations 

1. Data from **Laghouat, Algeria** (Feb–Aug 2022), not an Egyptian plant.
2. Dust series is partly **reconstructed** between sparse gravimetric samples.
3. Single **module-scale** measurements; plant-level kWh uses a user `plant_scale`.
4. Forecast residuals do **not** prove soiling as the cause of every drop.
5. Cleaning recommendations are **scenario-based**, not guaranteed savings.
6. No linkage in this module to thermal IR fault images (Feature 1).

---

## Example results

Results from the Laghouat chronological hold-out (module-scale watts).

**Validation (24h)**

| Model | MAE | RMSE | MAPE_day | R² |
|-------|-----|------|----------|-----|
| Ridge | 2.18 | 3.41 | 43.7 | 0.969 |
| Random Forest | 1.34 | 3.56 | 16.9 | 0.967 |
| LightGBM-Optuna | 1.25 | 3.26 | 15.4 | 0.972 |

**Test (24h)**

| Model | MAE | RMSE | MAPE_day | R² |
|-------|-----|------|----------|-----|
| Persistence | 1.04 | 3.27 | 12.0 | 0.978 |
| LightGBM-Optuna | 1.88 | 4.11 | 16.8 | 0.965 |

On this test window, persistence is strong due to daily seasonality; the learned model remains the deployment candidate after validation selection, with optional blending reported in training notebooks.

---

## Acknowledgments

- Laghouat PV + dust dataset authors and Mendeley release  
- SolarVision project scope: forecasting, anomaly support, and maintenance prioritization for PV operators
```
