#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
import matplotlib
matplotlib.use('Agg')  # ensure headless backend
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ---------------- CONFIG ----------------
OBS_DAYS = 14                 # past days
FC_HOURS = 48                 # forecast horizon
AMPLITUDES = [2.0, 1.0]      # meters for M2, S2 (larger for visibility)
PERIODS = [12.42, 12.00]     # hours
PHASES = [0, 0]              # initial phase
# ----------------------------------------

def generate_synthetic_tide(start, total_hours, amplitudes, periods, phases):
    hours = np.arange(total_hours)
    tide_values = np.zeros_like(hours, dtype=float)
    for A, T, phi in zip(amplitudes, periods, phases):
        tide_values += A * np.sin(2 * np.pi * hours / T + phi)
    times = pd.date_range(start=start, periods=total_hours, freq='h')
    return pd.Series(tide_values, index=times)

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=OBS_DAYS)

    total_hours = OBS_DAYS*24 + FC_HOURS

    # 1) Synthetic tide series
    tide_series = generate_synthetic_tide(start, total_hours, AMPLITUDES, PERIODS, PHASES)
    series_observed = tide_series[:OBS_DAYS*24]  # observed part

    # 2) Scale for ARIMA stability
    mean_obs = series_observed.mean()
    std_obs = series_observed.std()
    series_scaled = (series_observed - mean_obs) / std_obs

    # 3) ARIMA forecast
    try:
        model = ARIMA(series_scaled, order=(1,1,1)).fit()
        forecast_scaled = model.forecast(steps=FC_HOURS)
        forecast = forecast_scaled * std_obs + mean_obs
        forecast_times = pd.date_range(series_observed.index[-1] + timedelta(hours=1),
                                       periods=FC_HOURS, freq='h')
        forecast_series = pd.Series(forecast, index=forecast_times)
    except Exception as e:
        print(f"[ERROR] ARIMA forecast failed: {e}")
        forecast_series = tide_series[-FC_HOURS:]

    # 4) Combined forecast (no surge)
    combined = tide_series.reindex(forecast_series.index).ffill()

    # 5) Plot and save
    plt.figure(figsize=(12,6))
    series_observed[-7*24:].plot(label="Synthetic Observed (last 7 days)")
    forecast_series.plot(label="ARIMA Forecast")
    combined.plot(label="Synthetic Tide Forecast")
    plt.title("Laguna Water Level Forecast (Synthetic, Warning-Free)")
    plt.ylabel("Water level (m)")
    plt.xlabel("Time (UTC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("laguna_forecast.png")
    print("[INFO] Forecast plot saved to 'laguna_forecast.png'")

if __name__ == "__main__":
    main()
