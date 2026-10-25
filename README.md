# SolarVision — Time Series Module

Forecasting, anomaly alerts, and a transparent cleaning cost–benefit decision for solar PV monitoring.

**Live dashboard:** https://solarvisiontimeseriesfeature-gajz3vb3wa8kylffnrrceq.streamlit.app/

> Part of the SolarVision project. Thermal (IR) fault detection is a separate module and is **not** linked to this one.

---

## What it does

| Capability | Description |
|---|---|
| **Forecasting** | Predicts PV power 24 h ahead (a lighter 48 h experiment is included) |
| **Anomaly alerts** | Flags *sustained* daytime under-production versus the forecast |
| **Cleaning decision** | Scenario-based **Clean / Wait** from user-supplied price, cost and recovery inputs |
| **API + Dashboard** | FastAPI service and a Streamlit dashboard |

An alert means *"production is below the model's expectation"*. It is **not** proof of dust; shading, clouds or sensor faults can cause the same signal.

**Data source:** [Hourly PV Power and Dust Accumulation Dataset — Laghouat, Algeria (2022)](https://data.mendeley.com/datasets/4jdt83yy43). No external APIs and no invented sensors.

---

## Project structure

```
├── 01_preprocessing.ipynb        # Cleaning, features, targets, chronological split
├── 02_training.ipynb             # 24h models, MLflow, Optuna, SHAP, blend
├── 02b_training_48h.ipynb        # Light 48h-ahead experiment
├── 03_anomaly_cleaning.ipynb     # Alert rules + cleaning economics
├── api.py                        # FastAPI service
├── app_streamlit.py              # Streamlit dashboard
├── Laghouat_2022_final.csv       # Dataset
├── forecast_test_predictions_24h.csv   # Test predictions (dashboard input)
├── anomaly_results_24h.csv / anomaly_alerts_24h.csv
├── cleaning_decision_examples.csv
├── model_comparison_val_24h.csv / model_comparison_test_24h.csv / model_comparison_val_48h.csv
├── mlflow_solarvision.db         # MLflow tracking store
├── processed_data/               # Generated parquet + metadata
├── models/                       # Saved joblib models
└── figures/                      # SHAP and anomaly plots
```

---

## Quick start

```bash
pip install pandas numpy scikit-learn lightgbm optuna mlflow shap torch \
            pyarrow matplotlib joblib fastapi uvicorn streamlit pydantic
```
Python 3.10+ recommended.

### 1. Run the notebooks in order
```
01_preprocessing.ipynb → 02_training.ipynb → (02b_training_48h.ipynb) → 03_anomaly_cleaning.ipynb
```
Place `Laghouat_2022_final.csv` in the project root first.

### 2. Dashboard
```bash
streamlit run app_streamlit.py
```
Needs only `streamlit`, `pandas`, `numpy` and `forecast_test_predictions_24h.csv` in the same folder.

### 3. API
```bash
uvicorn api:app --reload --port 8000
```
Needs the `models/` folder produced by notebook 02. Docs: `http://localhost:8000/docs`.

| Endpoint | Method | Role |
|---|---|---|
| `/health` | GET | Model load status |
| `/forecast` | POST | Power prediction from a feature map (optional blend) |
| `/anomaly` | POST | Sustained-alert detection on actual vs predicted points |
| `/cleaning` | POST | Clean / Wait cost–benefit |
| `/demo/test_summary` | GET | Summary of the saved test predictions |

> Missing features in `/forecast` are filled with `0.0`, so send the full feature set.

### 4. MLflow UI (optional)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_solarvision.db
```

---

## Modeling approach

**Data:** 4,368 hourly rows (2022-02-17 → 2022-08-17); 4,272 remain after lag/target construction.

**Split (chronological, no shuffle, 70/15/15):**

| Set | Rows | Period |
|---|---|---|
| Train | 2,990 | 2022-02-20 → 2022-06-24 |
| Validation | 640 | 2022-06-24 → 2022-07-21 |
| Test | 642 | 2022-07-21 → 2022-08-16 |

**Target:** `P_dusty_W` shifted −24 h (main) and −48 h (experiment).

**Features (47 for 24 h):** calendar + cyclic encodings, power lags (1–72 h) and trailing rolling stats, PCHIP-reconstructed dust density + lags/rolling, irradiance/temperature history, weather fields, daytime flag. All lags and rolling windows use past values only; concurrent `P_dusty_W` and `P_clean_*` are excluded.

**Models compared:** Persistence (baseline), Ridge, Random Forest, LightGBM (default and Optuna-tuned, 25 trials, `TimeSeriesSplit`), and a hybrid LSTM. Experiments are tracked with MLflow; SHAP is used for explanations. Selection is by **validation MAE** among learned models. The deployed model is **Random Forest**.

**Metrics:** MAE, RMSE, daytime MAPE, R².

---

## Results (module-scale watts)

**Validation, 24 h**

| Model | MAE | RMSE | MAPE_day | R² |
|---|---|---|---|---|
| Persistence | 0.919 | 3.538 | 11.6% | 0.967 |
| **Random Forest** | 1.207 | 3.060 | 14.5% | 0.975 |
| LightGBM (default) | 1.228 | 3.137 | 14.9% | 0.974 |
| LSTM | 1.571 | 3.017 | 22.2% | 0.976 |
| LightGBM-Optuna | 1.599 | 3.200 | 20.4% | 0.973 |
| Ridge | 2.820 | 3.788 | 45.1% | 0.962 |

**Test, 24 h (opened once)**

| Model | MAE | RMSE | MAPE_day | R² |
|---|---|---|---|---|
| Persistence | **1.038** | 3.269 | 11.4% | 0.978 |
| Random Forest (deployed) | 1.744 | 3.748 | 15.3% | 0.971 |
| Blend (w = 0.20) | 1.068 | **3.135** | **11.2%** | **0.979** |

**Honest reading:** on this test window the Random Forest does **not** beat the persistence baseline (daily seasonality is very strong). A blend of 20% model + 80% persistence, with the weight chosen on validation, beats persistence on RMSE, MAPE and R² while being marginally worse on MAE. Both are reported.

**48 h experiment:** LightGBM was selected on validation (MAE 1.350) but on test it reaches MAE 1.907 versus 1.104 for persistence. It is not wired into the API or dashboard.

**Reference (Seghir et al., MATLAB, same site):** RF RMSE ≈ 4.46 W, R² ≈ 0.96 on a *nowcast* task. This is a methodological reference only, not a comparable 24 h-ahead benchmark.

---

## Anomaly & cleaning logic

**Anomaly:** a daytime hour is anomalous when forecast ≥ 5 W and actual ≤ forecast × (1 − 20%). A **sustained alert** is raised when ≥ 3 of the last 6 hours are anomalous.

On the real test window there are 10 anomalous hours and **0 sustained alerts**.

**Cleaning decision:**
```
net_benefit = (energy_loss_kwh × recoverable_fraction × electricity_price) − cleaning_cost
decision    = Clean if net_benefit > 0 else Wait
```
Price, cost, recovery fraction and `plant_scale` are user inputs. The notebook computes loss from the whole test window (expected 9.04 vs actual 9.85 module-kWh → loss 0 → **Wait**). The dashboard computes the hourly shortfall on days that have a sustained alert and only recommends **Clean** when an alert exists and the net benefit is positive.

### Dashboard demo
The test data contains no sustained soiling, so the dashboard offers a **simulated soiling event** (default: 35% power loss for 3 days from 2022-08-05, daytime hours only):

| Mode | Sustained alert hours | Decision |
|---|---|---|
| Simulation off (real data) | 0 | Wait |
| Simulation on (defaults) | 26 | Clean (net benefit ≈ 405.8) |

The simulation is for demonstration only and is labelled as such in the app.

---

## Limitations

1. Data come from **Laghouat, Algeria** (Feb–Aug 2022), one season, one site, not an Egyptian plant.
2. Single **module-scale** series; plant-level kWh uses a hypothetical `plant_scale`.
3. Dust density is **reconstructed** (PCHIP) from only 48 gravimetric measurements.
4. Features use current weather, not weather forecasts.
5. The learned model does not beat persistence on MAE for the test window; see Results.
6. The LSTM receives seven irregularly spaced lags, so it is not a true hourly-sequence model.
7. Forecast residuals do **not** prove soiling; alerts depend on threshold choices.
8. Cleaning recommendations are scenarios, not guaranteed savings.
9. No link to thermal/IR fault images.

---

## Acknowledgments

- Authors of the Laghouat PV + dust dataset and the Mendeley release.
- SolarVision project scope: forecasting, anomaly support and maintenance prioritisation for PV operators.
