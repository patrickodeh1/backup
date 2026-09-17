"""
generate_demo_data.py — fallback synthetic dataset + trained pipeline.

This is ONLY a stand-in for local testing before you point the backend at your
real MetroPT-3 CSV and your real trained model (models/compressor_pipeline_v3.pkl).

It builds a short, statistically faithful synthetic history: sawtooth duty-cycling
sensors with realistic jitter, plus one embedded failure window (continuous running,
flattened cycling, rising oil temp and motor current) matching the signature we
validated on the real data. It then trains a real IsolationForest on it using the
exact same feature engineering and pipeline classes used in production, so the
backend, replay, and dashboard can all be exercised end to end.

For the actual pitch: replace data/MetroPT3(AirCompressor).csv and
models/compressor_pipeline_v3.pkl with your real files and skip this script.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pipeline import CompressorAnomalyPipeline  # noqa: E402

SENSORS = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
FEATURE_COLS = [
    'H1_roll_std', 'H1_cycle_transitions',
    'TP2_roll_std', 'TP2_roll_range',
    'TP3_roll_std', 'TP3_roll_range',
    'Reservoirs_roll_std', 'Reservoirs_roll_range',
    'Oil_temperature_roll_mean', 'Oil_temperature_roll_trend',
    'Motor_current_roll_std', 'Motor_current_roll_range',
]
ROLLING_WINDOW = 60
H1_LOW_THRESHOLD = 1.0


def engineer_features(df):
    for s in SENSORS:
        df[f'{s}_roll_std'] = df[s].rolling(ROLLING_WINDOW, min_periods=1).std()
        df[f'{s}_roll_range'] = (
            df[s].rolling(ROLLING_WINDOW, min_periods=1).max()
            - df[s].rolling(ROLLING_WINDOW, min_periods=1).min()
        )
        df[f'{s}_roll_mean'] = df[s].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['H1_low'] = (df['H1'] < H1_LOW_THRESHOLD).astype(int)
    df['H1_cycle_transitions'] = df['H1_low'].diff().abs().rolling(
        ROLLING_WINDOW, min_periods=1).sum()
    for s in ['Oil_temperature', 'Motor_current', 'TP3', 'Reservoirs']:
        df[f'{s}_diff'] = df[s].diff()
        df[f'{s}_roll_trend'] = df[s].rolling(ROLLING_WINDOW, min_periods=1).mean().diff()
    return df


def _sawtooth_cycle(n, period_low=100, period_high=140, low=8.1, high=8.9, noise=0.03, rng=None):
    """Duty-cycling pressure signal: climbs, cuts off, drifts down, repeats -- with jitter."""
    rng = rng or np.random.default_rng(0)
    vals = np.zeros(n)
    i = 0
    level = low
    rising = True
    while i < n:
        period = rng.integers(period_low, period_high)
        target = high if rising else low
        seg = np.linspace(level, target, period)
        seg = seg + rng.normal(0, noise, size=period)
        end = min(i + period, n)
        vals[i:end] = seg[:end - i]
        level = vals[end - 1]
        rising = not rising
        i = end
    return vals


def generate(n_rows=5400, rng_seed=42):
    """~15 hours at 10s sampling, with one ~40 minute failure window near the end."""
    rng = np.random.default_rng(rng_seed)
    timestamps = pd.date_range('2026-09-10 00:00:00', periods=n_rows, freq='10s')

    H1 = _sawtooth_cycle(n_rows, low=0.02, high=8.2, noise=0.05, rng=rng)
    TP2 = _sawtooth_cycle(n_rows, low=8.0, high=9.2, noise=0.04, rng=rng)
    TP3 = TP2 + rng.normal(0, 0.05, n_rows) - 0.1
    Reservoirs = TP2 + rng.normal(0, 0.05, n_rows) - 0.05
    DV_pressure = np.clip(rng.normal(1.0, 0.15, n_rows), 0, None)
    Oil_temperature = 65 + 0.002 * np.arange(n_rows) % 20 + rng.normal(0, 0.8, n_rows)
    Motor_current = 4.2 + 0.6 * (H1 > 1.0).astype(float) + rng.normal(0, 0.15, n_rows)

    # Embed a failure window: continuous running -> flat H1 near cutoff, rising oil temp/current
    fail_start = int(n_rows * 0.78)
    fail_len = 240  # 40 minutes at 10s sampling
    fail_end = fail_start + fail_len
    ramp = np.linspace(0, 1, fail_len)
    H1[fail_start:fail_end] = 8.3 + rng.normal(0, 0.03, fail_len)
    TP2[fail_start:fail_end] = 8.9 + 0.15 * ramp + rng.normal(0, 0.03, fail_len)
    TP3[fail_start:fail_end] = TP2[fail_start:fail_end] - 0.08 + rng.normal(0, 0.03, fail_len)
    Reservoirs[fail_start:fail_end] = TP2[fail_start:fail_end] - 0.05 + rng.normal(0, 0.03, fail_len)
    Oil_temperature[fail_start:fail_end] = Oil_temperature[fail_start] + 12 * ramp + rng.normal(0, 0.5, fail_len)
    Motor_current[fail_start:fail_end] = 5.4 + 0.8 * ramp + rng.normal(0, 0.1, fail_len)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'TP2': TP2, 'TP3': TP3, 'H1': H1, 'DV_pressure': DV_pressure,
        'Reservoirs': Reservoirs, 'Oil_temperature': Oil_temperature,
        'Motor_current': Motor_current,
    })
    df['label'] = 0
    df.loc[fail_start:fail_end - 1, 'label'] = 1
    df.attrs['failure_start'] = str(timestamps[fail_start])
    df.attrs['failure_end'] = str(timestamps[fail_end - 1])
    return df


def build_and_save(data_out, model_out):
    df = generate()
    df = engineer_features(df)

    train = df[df['label'] == 0].copy()
    X_train = train[FEATURE_COLS].dropna()
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    model.fit(X_train)

    stats = X_train.agg(['mean', 'std']).T
    stats.columns = ['normal_mean', 'normal_std']

    pipeline = CompressorAnomalyPipeline(
        model=model, normal_stats=stats, feature_cols=FEATURE_COLS,
        threshold_percentile=5, persistence_window=ROLLING_WINDOW, persistence_ratio=0.5,
    )
    scores = model.decision_function(df[FEATURE_COLS].dropna())
    pipeline.set_threshold(scores)

    os.makedirs(os.path.dirname(data_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    df.to_csv(data_out, index=False)
    joblib.dump(pipeline, model_out)
    print(f"Demo data saved to {data_out} ({len(df)} rows)")
    print(f"Demo model saved to {model_out}")
    print(f"Embedded failure window: {df.attrs['failure_start']} -> {df.attrs['failure_end']}")
    return df, pipeline


if __name__ == "__main__":
    build_and_save(
        data_out=os.path.join(os.path.dirname(__file__), '..', 'data', 'demo_compressor_data.csv'),
        model_out=os.path.join(os.path.dirname(__file__), '..', 'models', 'demo_pipeline.pkl'),
    )
