"""
SolarVision — Time Series dashboard (single page, demo-ready)

Run:
  streamlit run app_streamlit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SolarVision Time Series", layout="wide")

PRED_PATH = Path("forecast_test_predictions_24h.csv")

st.title("SolarVision — Is my panel underperforming? Should I clean it?")
st.caption(
    "For plant operators / maintenance managers. Laghouat module-scale data only. "
    "An alert means 'below forecast', not a proven dust cause."
)

with st.expander("How this page works (3 steps)", expanded=True):
    st.markdown(
        """
1. **Forecast** — the model predicts the expected power for every hour.
2. **Anomaly alert** — if actual power stays ≥ X% below the forecast for several hours in a row, we raise a *sustained alert*.
3. **Cleaning decision** — we turn the lost energy into money and compare it with the cleaning cost → **Clean** or **Wait**.
"""
    )

if not PRED_PATH.exists():
    st.error(
        f"`{PRED_PATH}` not found. Put it next to this file "
        "(it is produced by notebook 02)."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
raw = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True).sort_index()
t_min, t_max = raw.index.min().date(), raw.index.max().date()

st.sidebar.header("1) Forecast & alert rule")
fc_options = ["Model (Random Forest)"]
if "predicted_blend_W" in raw.columns:
    fc_options = fc_options + ["Blend (model + persistence)"]
fc_choice = st.sidebar.selectbox("Forecast used as 'expected'", fc_options)
drop_pct = st.sidebar.slider("Alert if actual is below forecast by (%)", 5, 50, 20) / 100.0

st.sidebar.header("2) Demo: simulate a dirty panel")
simulate = st.sidebar.checkbox("Simulate soiling event", value=True)
fault_drop = st.sidebar.slider("Power loss during event (%)", 5, 60, 35) / 100.0
default_start = min(max(pd.Timestamp("2022-08-05").date(), t_min), t_max)
fault_start = st.sidebar.date_input(
    "Event start", value=default_start, min_value=t_min, max_value=t_max
)
fault_days = st.sidebar.slider("Event length (days)", 1, 7, 3)

st.sidebar.header("3) Cleaning economics (scenario)")
price = st.sidebar.number_input("Electricity price / kWh", value=2.0, min_value=0.0)
cost = st.sidebar.number_input("Cleaning cost", value=50.0, min_value=0.0)
recover = st.sidebar.slider("Recoverable fraction", 0.0, 1.0, 0.80, 0.05)
scale = st.sidebar.number_input("Plant scale factor (× one module)", value=1000.0, min_value=1.0)

# --------------------------------------------------------------------------
# Build working table
# --------------------------------------------------------------------------
df = raw.copy()
fcol = "predicted_blend_W" if fc_choice.startswith("Blend") else "predicted_power_W"
df["forecast"] = df[fcol].astype(float)
df["actual_used"] = df["actual_power_W"].astype(float)
day = (
    df["is_daytime"].astype(bool)
    if "is_daytime" in df.columns
    else df["actual_power_W"] > 0.5
)

fault_mask = pd.Series(False, index=df.index)
if simulate:
    start = pd.Timestamp(fault_start)
    end = start + pd.Timedelta(days=fault_days)
    fault_mask = (df.index >= start) & (df.index < end) & day
    df.loc[fault_mask, "actual_used"] *= 1.0 - fault_drop

under = day & (df["forecast"] >= 5) & (
    df["actual_used"] <= df["forecast"] * (1 - drop_pct)
)
df["is_anomaly_hour"] = under.astype(int)
df["anomaly_count_6h"] = df["is_anomaly_hour"].rolling(6, min_periods=1).sum()
df["anomaly_alert"] = (df["anomaly_count_6h"] >= 3).astype(int)

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
mae = np.mean(np.abs(df["actual_power_W"] - df["predicted_power_W"]))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast MAE (W, test)", f"{mae:.2f}")
c2.metric("Hours below forecast", int(df["is_anomaly_hour"].sum()))
c3.metric("Sustained alert hours", int(df["anomaly_alert"].sum()))
c4.metric("Simulated event", "ON" if simulate else "OFF")

# --------------------------------------------------------------------------
# Chart (Streamlit native — no matplotlib)
# --------------------------------------------------------------------------
st.subheader("Actual vs forecast")
chart_df = pd.DataFrame(
    {
        "Forecast (expected)": df["forecast"],
        "Actual": df["actual_used"],
        "Alert": df["actual_used"].where(df["anomaly_alert"] == 1),
    },
    index=df.index,
)
st.line_chart(chart_df, height=280, color=["#1f77b4", "#ff7f0e", "#d62728"])

al = df[df["anomaly_alert"] == 1]
if len(al):
    st.caption(
        f"Sustained alerts: **{len(al)}** hours "
        f"(first: {al.index.min()}, last: {al.index.max()})"
    )
if simulate:
    st.caption(
        f"Simulated soiling window: **{fault_start}** → "
        f"**{(pd.Timestamp(fault_start) + pd.Timedelta(days=fault_days)).date()}** "
        f"({int(fault_drop * 100)}% power loss on daytime hours)"
    )
# --------------------------------------------------------------------------
# Cleaning decision
# --------------------------------------------------------------------------
st.subheader("Cleaning decision")

if len(al):
    alert_days = set(al.index.normalize())
    in_alert_days = day & df.index.normalize().isin(alert_days)
    shortfall = (df["forecast"] - df["actual_used"]).clip(lower=0)[in_alert_days]
    module_loss_kwh = float(shortfall.sum() / 1000.0)
    module_expected_kwh = float(df.loc[in_alert_days, "forecast"].sum() / 1000.0)
    module_actual_kwh = float(df.loc[in_alert_days, "actual_used"].sum() / 1000.0)
else:
    alert_days = set()
    module_loss_kwh = module_expected_kwh = module_actual_kwh = 0.0

loss = module_loss_kwh * scale
recovered = loss * recover
revenue = recovered * price
net = revenue - cost
decision = "Clean" if (len(al) and net > 0) else "Wait"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated loss (plant kWh)", f"{loss:,.1f}")
k2.metric("Recoverable revenue", f"{revenue:,.1f}")
k3.metric("Net benefit", f"{net:,.1f}")
k4.metric("Decision", decision)

if not len(al):
    st.info("No sustained alert → nothing to recover → **Wait**.")
elif decision == "Clean":
    st.success(
        f"Sustained under-production detected. Expected recovery ({revenue:,.0f}) "
        f"is higher than cleaning cost ({cost:,.0f}) → **Clean**."
    )
else:
    st.warning(
        f"Alert exists but recovery ({revenue:,.0f}) does not cover cleaning cost "
        f"({cost:,.0f}) → **Wait**."
    )

with st.expander("Calculation details"):
    st.write(
        {
            "alert_days": sorted(str(d.date()) for d in alert_days),
            "module_expected_kwh_on_alert_days": round(module_expected_kwh, 4),
            "module_actual_kwh_on_alert_days": round(module_actual_kwh, 4),
            "module_shortfall_kwh": round(module_loss_kwh, 4),
            "plant_scale": scale,
            "recoverable_fraction": recover,
            "electricity_price": price,
            "cleaning_cost": cost,
        }
    )

if len(al):
    st.subheader("Alert log")
    st.dataframe(
        df.loc[df["anomaly_alert"] == 1, ["actual_used", "forecast", "anomaly_count_6h"]]
        .rename(columns={"actual_used": "actual_W", "forecast": "forecast_W"})
        .round(2),
    )

st.markdown(
    """
---
**Limits**
- Data: Laghouat single-module series (not a full plant); scaling is hypothetical.
- The soiling event is **simulated** for the demo; with the default model forecast the real test data has no sustained alert.
- Alert = below forecast; shading, clouds or sensor faults can also cause it.
- Financial values are scenarios from the sidebar inputs.
"""
)
