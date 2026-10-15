
#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning


DEFAULT_DAYS = 14
DEFAULT_HOURS = 48
DEFAULT_AMPLITUDES = [2.0, 1.0]
DEFAULT_PERIODS = [12.42, 12.0]
DEFAULT_PHASES = [0.0, 0.0]
DEFAULT_VALIDATION_HOURS = 48
DEFAULT_OUTPUT = "laguna_forecast"

METHODS = ("harmonic", "arima", "hybrid")


@dataclass
class ForecastResult:
    method: str
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class ModelMetrics:
    method: str
    mae: float
    rmse: float
    bias: float
    residual_std: float
    samples: int


def parse_float_list(value):
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated numeric values"
        ) from exc

    if not values or not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "Values must be finite numbers"
        )

    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Laguna CLI tidal forecasting and model validation"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS
    )

    parser.add_argument(
        "--validation-hours",
        type=int,
        default=DEFAULT_VALIDATION_HOURS
    )

    parser.add_argument(
        "--amplitudes",
        type=parse_float_list,
        default=DEFAULT_AMPLITUDES
    )

    parser.add_argument(
        "--periods",
        type=parse_float_list,
        default=DEFAULT_PERIODS
    )

    parser.add_argument(
        "--phases",
        type=parse_float_list,
        default=DEFAULT_PHASES
    )

    parser.add_argument(
        "--method",
        choices=("auto",) + METHODS,
        default="auto"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--json",
        action="store_true"
    )

    parser.add_argument(
        "--quiet",
        action="store_true"
    )

    return parser.parse_args()


def validate_args(args):
    if args.days <= 0:
        raise ValueError("Observation days must be positive")

    if args.hours <= 0:
        raise ValueError("Forecast hours must be positive")

    if args.validation_hours <= 0:
        raise ValueError("Validation hours must be positive")

    if len(args.amplitudes) != len(args.periods):
        raise ValueError("Amplitude and period counts must match")

    if len(args.periods) != len(args.phases):
        raise ValueError("Period and phase counts must match")

    if not args.amplitudes:
        raise ValueError("At least one tidal component is required")

    if any(value <= 0 for value in args.amplitudes):
        raise ValueError("Amplitudes must be positive")

    if any(value <= 0 for value in args.periods):
        raise ValueError("Periods must be positive")

    observation_hours = args.days * 24

    if observation_hours <= args.validation_hours + 24:
        raise ValueError(
            "Observation window must exceed validation window by 24 hours"
        )


def generate_tide(hours, amplitudes, periods, phases):
    hours = np.asarray(hours, dtype=float)
    values = np.zeros(len(hours), dtype=float)

    for amplitude, period, phase in zip(
        amplitudes,
        periods,
        phases
    ):
        values += amplitude * np.sin(
            2.0 * np.pi * hours / period + phase
        )

    return values


def harmonic_matrix(hours, periods):
    hours = np.asarray(hours, dtype=float)

    columns = [np.ones(len(hours), dtype=float)]

    for period in periods:
        angle = 2.0 * np.pi * hours / period
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))

    return np.column_stack(columns)


def harmonic_forecast(observed, horizon, periods):
    count = len(observed)

    training_hours = np.arange(count)
    future_hours = np.arange(count, count + horizon)

    design = harmonic_matrix(training_hours, periods)

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        observed,
        rcond=None
    )

    fitted = design @ coefficients

    future_design = harmonic_matrix(
        future_hours,
        periods
    )

    predicted = future_design @ coefficients

    return predicted, observed - fitted


def arima_forecast(observed, horizon):
    observed = np.asarray(observed, dtype=float)

    mean = float(np.mean(observed))
    scale = float(np.std(observed))

    if scale <= 1e-12:
        raise ValueError("ARIMA requires nonconstant observations")

    scaled = (observed - mean) / scale

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)

        model = ARIMA(
            scaled,
            order=(2, 0, 2),
            enforce_stationarity=True,
            enforce_invertibility=True
        )

        fitted = model.fit()

    convergence_warnings = [
        warning
        for warning in captured
        if issubclass(warning.category, ConvergenceWarning)
    ]

    if convergence_warnings:
        raise RuntimeError("ARIMA optimization did not converge")

    if not fitted.mle_retvals.get("converged", True):
        raise RuntimeError("ARIMA optimization did not converge")

    forecast = np.asarray(
        fitted.forecast(steps=horizon),
        dtype=float
    )

    predicted = forecast * scale + mean

    residuals = np.asarray(
        fitted.resid,
        dtype=float
    ) * scale

    return predicted, residuals


def hybrid_forecast(observed, horizon, periods):
    harmonic_values, harmonic_residuals = harmonic_forecast(
        observed,
        horizon,
        periods
    )

    if np.std(harmonic_residuals) <= 1e-10:
        return harmonic_values, harmonic_residuals

    residual_forecast, residual_errors = arima_forecast(
        harmonic_residuals,
        horizon
    )

    return harmonic_values + residual_forecast, residual_errors


def forecast_method(method, observed, horizon, periods):
    if method == "harmonic":
        return harmonic_forecast(
            observed,
            horizon,
            periods
        )

    if method == "arima":
        return arima_forecast(
            observed,
            horizon
        )

    if method == "hybrid":
        return hybrid_forecast(
            observed,
            horizon,
            periods
        )

    raise ValueError(f"Unknown forecasting method: {method}")


def calculate_metrics(method, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ")

    if not np.all(np.isfinite(predicted)):
        raise ValueError("Forecast contains nonfinite values")

    errors = actual - predicted

    return ModelMetrics(
        method=method,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        bias=float(np.mean(errors)),
        residual_std=float(np.std(errors, ddof=1))
        if len(errors) > 1 else 0.0,
        samples=len(errors)
    )


def evaluate_models(observed, validation_hours, periods):
    training = observed[:-validation_hours]
    actual = observed[-validation_hours:]

    metrics = []
    failures = {}

    for method in METHODS:
        try:
            predicted, _ = forecast_method(
                method,
                training,
                validation_hours,
                periods
            )

            result = calculate_metrics(
                method,
                actual,
                predicted
            )

            metrics.append(result)

        except Exception as exc:
            failures[method] = str(exc)
            logging.warning(
                "Validation failed for %s: %s",
                method,
                exc
            )

    metrics.sort(
        key=lambda item: (item.rmse, item.mae)
    )

    return metrics, failures


def select_method(requested, metrics):
    available = {item.method for item in metrics}

    if requested == "auto":
        if not metrics:
            raise RuntimeError(
                "No forecasting models passed validation"
            )

        return metrics[0].method

    if requested not in available:
        raise RuntimeError(
            f"Requested model {requested} did not pass validation"
        )

    return requested


def build_intervals(predicted, residual_std):
    predicted = np.asarray(predicted, dtype=float)

    margin = 1.96 * residual_std

    return predicted - margin, predicted + margin


def create_forecast(
    method,
    observed,
    horizon,
    periods,
    metrics
):
    predicted, _ = forecast_method(
        method,
        observed,
        horizon,
        periods
    )

    metric = next(
        item for item in metrics
        if item.method == method
    )

    lower, upper = build_intervals(
        predicted,
        metric.residual_std
    )

    return ForecastResult(
        method=method,
        values=predicted,
        lower=lower,
        upper=upper
    )


def write_csv(path, headers, rows):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def save_forecast(path, times, result, actual):
    rows = []

    for timestamp, predicted, lower, upper, reference in zip(
        times,
        result.values,
        result.lower,
        result.upper,
        actual
    ):
        rows.append([
            timestamp.isoformat(),
            result.method,
            float(predicted),
            float(lower),
            float(upper),
            float(reference),
            float(reference - predicted)
        ])

    write_csv(
        path,
        [
            "timestamp_utc",
            "method",
            "forecast_m",
            "lower_95_m",
            "upper_95_m",
            "synthetic_reference_m",
            "error_m"
        ],
        rows
    )


def save_metrics(path, metrics):
    rows = [
        [
            item.method,
            item.mae,
            item.rmse,
            item.bias,
            item.residual_std,
            item.samples
        ]
        for item in metrics
    ]

    write_csv(
        path,
        [
            "method",
            "mae",
            "rmse",
            "bias",
            "residual_std",
            "samples"
        ],
        rows
    )


def save_validation(path, observed, validation_hours, periods, times):
    training = observed[:-validation_hours]
    actual = observed[-validation_hours:]

    rows = []

    for method in METHODS:
        try:
            predicted, _ = forecast_method(
                method,
                training,
                validation_hours,
                periods
            )

            for timestamp, reference, value in zip(
                times,
                actual,
                predicted
            ):
                rows.append([
                    timestamp.isoformat(),
                    method,
                    float(reference),
                    float(value),
                    float(reference - value)
                ])

        except Exception as exc:
            logging.warning(
                "Skipping validation export for %s: %s",
                method,
                exc
            )

    write_csv(
        path,
        [
            "timestamp_utc",
            "method",
            "actual_m",
            "predicted_m",
            "error_m"
        ],
        rows
    )


def print_table(headers, rows):
    rows = [
        [str(value) for value in row]
        for row in rows
    ]

    widths = [
        max(
            len(str(header)),
            max(
                (len(row[index]) for row in rows),
                default=0
            )
        )
        for index, header in enumerate(headers)
    ]

    pattern = "  ".join(
        f"{{:<{width}}}"
        for width in widths
    )

    print(pattern.format(*headers))
    print(pattern.format(
        *["-" * width for width in widths]
    ))

    for row in rows:
        print(pattern.format(*row))


def print_report(args, metrics, result, times, actual):
    print()
    print("Laguna Tide Forecast")
    print("=" * 72)
    print(f"Observation window : {args.days} days")
    print(f"Forecast horizon   : {args.hours} hours")
    print(f"Validation window : {args.validation_hours} hours")
    print(f"Selected model     : {result.method}")
    print()

    print("Model Validation")
    print("-" * 72)

    print_table(
        ["Model", "MAE", "RMSE", "Bias", "Samples"],
        [
            [
                item.method,
                f"{item.mae:.6f}",
                f"{item.rmse:.6f}",
                f"{item.bias:.6f}",
                item.samples
            ]
            for item in metrics
        ]
    )

    print()
    print("Forecast Preview")
    print("-" * 72)

    preview = min(12, len(result.values))

    print_table(
        ["UTC Time", "Forecast", "Lower 95%", "Upper 95%", "Reference"],
        [
            [
                times[index].strftime("%Y-%m-%d %H:%M"),
                f"{result.values[index]:.4f}",
                f"{result.lower[index]:.4f}",
                f"{result.upper[index]:.4f}",
                f"{actual[index]:.4f}"
            ]
            for index in range(preview)
        ]
    )

    print()


def build_json_report(args, metrics, result, times, actual, failures):
    return {
        "tool": "Laguna",
        "version": "2.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "observation_days": args.days,
        "forecast_hours": args.hours,
        "validation_hours": args.validation_hours,
        "selected_method": result.method,
        "metrics": [
            {
                "method": item.method,
                "mae": item.mae,
                "rmse": item.rmse,
                "bias": item.bias,
                "residual_std": item.residual_std,
                "samples": item.samples
            }
            for item in metrics
        ],
        "failures": failures,
        "forecast": [
            {
                "timestamp_utc": timestamp.isoformat(),
                "forecast_m": float(predicted),
                "lower_95_m": float(lower),
                "upper_95_m": float(upper),
                "synthetic_reference_m": float(reference)
            }
            for timestamp, predicted, lower, upper, reference in zip(
                times,
                result.values,
                result.lower,
                result.upper,
                actual
            )
        ]
    }


def run(args):
    validate_args(args)

    observation_hours = args.days * 24
    total_hours = observation_hours + args.hours

    end = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0
    )

    start = end - timedelta(
        hours=observation_hours - 1
    )

    all_times = pd.date_range(
        start=start,
        periods=total_hours,
        freq="h",
        tz="UTC"
    )

    tide_values = generate_tide(
        np.arange(total_hours),
        args.amplitudes,
        args.periods,
        args.phases
    )

    observed = tide_values[:observation_hours]
    reference = tide_values[observation_hours:]

    metrics, failures = evaluate_models(
        observed,
        args.validation_hours,
        args.periods
    )

    selected = select_method(
        args.method,
        metrics
    )

    result = create_forecast(
        selected,
        observed,
        args.hours,
        args.periods,
        metrics
    )

    forecast_times = all_times[observation_hours:]

    prefix = Path(args.output)

    forecast_path = f"{prefix}.csv"
    metrics_path = f"{prefix}_metrics.csv"
    validation_path = f"{prefix}_validation.csv"

    save_forecast(
        forecast_path,
        forecast_times,
        result,
        reference
    )

    save_metrics(
        metrics_path,
        metrics
    )

    save_validation(
        validation_path,
        observed,
        args.validation_hours,
        args.periods,
        all_times[
            observation_hours - args.validation_hours:
            observation_hours
        ]
    )

    if args.json:
        report = build_json_report(
            args,
            metrics,
            result,
            forecast_times,
            reference,
            failures
        )

        print(json.dumps(
            report,
            indent=2,
            allow_nan=False
        ))

    elif not args.quiet:
        print_report(
            args,
            metrics,
            result,
            forecast_times,
            reference
        )

        print("Output Files")
        print("-" * 72)
        print(f"Forecast   : {forecast_path}")
        print(f"Metrics    : {metrics_path}")
        print(f"Validation : {validation_path}")
        print()

    logging.info(
        "Laguna completed with model %s",
        selected
    )


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.ERROR if args.quiet or args.json else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr
    )

    try:
        run(args)
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
