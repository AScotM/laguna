#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
import matplotlib
matplotlib.use('Agg')  # Ensure headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA

# ---------------- CONFIG ----------------
OBS_DAYS = 14                 # Past days to observe
FC_HOURS = 48                 # Forecast horizon in hours
AMPLITUDES = [2.0, 1.0]      # Meters for M2, S2 tides (larger for visibility)
PERIODS = [12.42, 12.00]     # Hours for M2, S2 tidal periods
PHASES = [0, 0]              # Initial phase for tidal components
# ----------------------------------------

def generate_synthetic_tide(start, total_hours, amplitudes, periods, phases):
    """
    Generate a synthetic tide series using sinusoidal components.

    Args:
        start (datetime): Start time for the series.
        total_hours (int): Total hours to generate.
        amplitudes (list): Amplitudes of tidal components (meters).
        periods (list): Periods of tidal components (hours).
        phases (list): Phase shifts of tidal components (radians).

    Returns:
        pd.Series: Synthetic tide series with datetime index.
    """
    hours = np.arange(total_hours)
    tide_values = np.zeros_like(hours, dtype=float)
    for A, T, phi in zip(amplitudes, periods, phases):
        tide_values += A * np.sin(2 * np.pi * hours / T + phi)
    times = pd.date_range(start=start, periods=total_hours, freq='h')
    return pd.Series(tide_values, index=times)

def main():
    """
    Generate synthetic tide data, forecast using ARIMA, and visualize/save results.
    Saves a plot and CSV file with the forecast.
    """
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Validate configuration
    if OBS_DAYS <= 0 or FC_HOURS <= 0:
        raise ValueError("OBS_DAYS and FC_HOURS must be positive integers")
    if not all(a > 0 for a in AMPLITUDES) or not all(p > 0 for p in PERIODS):
        raise ValueError("AMPLITUDES and PERIODS must be positive")

    # Set time range
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=OBS_DAYS)
    total_hours = OBS_DAYS * 24 + FC_HOURS

    # Generate synthetic tide series
    tide_series = generate_synthetic_tide(start, total_hours, AMPLITUDES, PERIODS, PHASES)
    series_observed = tide_series[:OBS_DAYS*24]  # Observed portion

    # Scale data for ARIMA stability
    mean_obs = series_observed.mean()
    std_obs = series_observed.std()
    if std_obs == 0:
        raise ValueError("Observed data has zero variance, cannot scale for ARIMA")
    series_scaled = (series_observed - mean_obs) / std_obs

    # ARIMA forecast
    try:
        # Note: ARIMA(1,1,1) is used; consider auto_arima or ACF/PACF for optimal order
        model = ARIMA(series_scaled, order=(1,1,1)).fit()
        forecast_scaled = model.forecast(steps=FC_HOURS)
        forecast = forecast_scaled * std_obs + mean_obs
        forecast_times = pd.date_range(series_observed.index[-1] + timedelta(hours=1),
                                       periods=FC_HOURS, freq='h')
        forecast_series = pd.Series(forecast, index=forecast_times)
    except Exception as e:
        print(f"[WARNING] ARIMA forecast failed: {e}. Using synthetic tide as fallback")
        forecast_series = tide_series[-FC_HOURS:]

    # Save forecast data to CSV
    forecast_series.to_csv("laguna_forecast.csv")
    print("[INFO] Forecast data saved to 'laguna_forecast.csv'")

    # Plotting
    plt.figure(figsize=(12,6))
    series_observed[-7*24:].plot(label="Synthetic Observed (last 7 days)")
    forecast_series.plot(label="ARIMA Forecast")
    plt.grid(True)
    plt.title("Laguna Water Level Forecast (Synthetic)")
    plt.ylabel("Water Level (m)")
    plt.xlabel("Time (UTC)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("laguna_forecast.png")
    print("[INFO] Forecast plot saved to 'laguna_forecast.png'")

if __name__ == "__main__":
    main()
