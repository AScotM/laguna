#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime, timedelta, timezone
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA

# ---------------- DEFAULT CONFIG ----------------
DEFAULT_OBS_DAYS = 14
DEFAULT_FC_HOURS = 48
DEFAULT_AMPLITUDES = [2.0, 1.0]   # meters
DEFAULT_PERIODS = [12.42, 12.00]  # hours
DEFAULT_PHASES = [0, 0]           # radians
# ------------------------------------------------


def generate_synthetic_tide(start, total_hours, amplitudes, periods, phases):
    """Generate a synthetic tide series using sinusoidal components."""
    hours = np.arange(total_hours)
    tide_values = np.zeros(total_hours, dtype=float)
    for A, T, phi in zip(amplitudes, periods, phases):
        tide_values += A * np.sin(2 * np.pi * hours / T + phi)
    times = pd.date_range(start=start, periods=total_hours, freq="h")
    return pd.Series(tide_values, index=times)


def run_forecast(obs_days, fc_hours, amplitudes, periods, phases, output_prefix):
    """Generate synthetic tide, forecast with ARIMA, save CSV and plot."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Validate inputs
    if obs_days <= 0 or fc_hours <= 0:
        raise ValueError("OBS_DAYS and FC_HOURS must be positive integers")
    if not all(a > 0 for a in amplitudes) or not all(p > 0 for p in periods):
        raise ValueError("AMPLITUDES and PERIODS must be positive")

    # Time range
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=obs_days)
    total_hours = obs_days * 24 + fc_hours

    # Synthetic tide
    tide_series = generate_synthetic_tide(start, total_hours, amplitudes, periods, phases)
    series_observed = tide_series[:obs_days * 24]

    # Scale for ARIMA stability
    mean_obs, std_obs = series_observed.mean(), series_observed.std()
    if std_obs == 0:
        raise ValueError("Observed data has zero variance, cannot scale for ARIMA")
    series_scaled = (series_observed - mean_obs) / std_obs

    # Forecast
    try:
        model = ARIMA(series_scaled, order=(1, 1, 1)).fit()
        forecast_scaled = model.forecast(steps=fc_hours)
        forecast = forecast_scaled * std_obs + mean_obs
        forecast_times = pd.date_range(series_observed.index[-1] + timedelta(hours=1),
                                       periods=fc_hours, freq="h")
        forecast_series = pd.Series(forecast, index=forecast_times)
        logging.info("ARIMA forecast completed successfully")
    except Exception as e:
        logging.warning(f"ARIMA forecast failed: {e}. Falling back to synthetic tide")
        forecast_series = tide_series[-fc_hours:]

    # Save forecast data
    csv_file = f"{output_prefix}.csv"
    forecast_series.to_csv(csv_file)
    logging.info(f"Forecast data saved to {csv_file}")

    # Plot results
    plt.figure(figsize=(12, 6))
    series_observed[-7 * 24:].plot(label="Observed (last 7 days)")
    forecast_series.plot(label="Forecast")
    plt.grid(True)
    plt.title("Laguna Water Level Forecast (Synthetic)")
    plt.ylabel("Water Level (m)")
    plt.xlabel("Time (UTC)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.legend()
    plt.tight_layout()
    png_file = f"{output_prefix}.png"
    plt.savefig(png_file)
    logging.info(f"Forecast plot saved to {png_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Laguna Tide Forecast Tool")
    parser.add_argument("--days", type=int, default=DEFAULT_OBS_DAYS,
                        help="Number of past days to observe (default: 14)")
    parser.add_argument("--hours", type=int, default=DEFAULT_FC_HOURS,
                        help="Forecast horizon in hours (default: 48)")
    parser.add_argument("--output", type=str, default="laguna_forecast",
                        help="Output file prefix (default: laguna_forecast)")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    logging.info(f"Starting forecast with {args.days} days observed, {args.hours}h forecast")

    run_forecast(
        obs_days=args.days,
        fc_hours=args.hours,
        amplitudes=DEFAULT_AMPLITUDES,
        periods=DEFAULT_PERIODS,
        phases=DEFAULT_PHASES,
        output_prefix=args.output,
    )


if __name__ == "__main__":
    main()
