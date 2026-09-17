"""
train.py — End-to-end training script for the compressor anomaly detection pipeline.

Run this to reproduce the full pipeline from raw MetroPT-3 data to a saved,
ready-to-use CompressorAnomalyPipeline object.

Usage:
    python src/train.py --data data/MetroPT3(AirCompressor).csv --out models/compressor_pipeline_v3.pkl

This script does NOT do exploratory analysis or plotting -- that lives in
notebooks/eda.ipynb. This is the reproducible, production-facing path:
raw data in, trained pipeline out.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from pipeline import CompressorAnomalyPipeline


# Documented failure windows (MetroPT-3, 2020, UCI dataset #791).
# Source: UCI dataset page / maintenance reports. See README for details.
FAILURES = [
    ('2020-04-18 00:00', '2020-04-18 23:59', 'Failure 1'),
    ('2020-05-29 23:30', '2020-05-30 06:00', 'Failure 2'),
    ('2020-06-05 10:00', '2020-06-07 14:30', 'Failure 3'),
    ('2020-07-15 14:30', '2020-07-15 19:00', 'Failure 4'),
]

SENSORS = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']

FEATURE_COLS = [
    'H1_roll_std', 'H1_cycle_transitions',
    'TP2_roll_std', 'TP2_roll_range',
    'TP3_roll_std', 'TP3_roll_range',
    'Reservoirs_roll_std', 'Reservoirs_roll_range',
    'Oil_temperature_roll_mean', 'Oil_temperature_roll_trend',
    'Motor_current_roll_std', 'Motor_current_roll_range',
]

ROLLING_WINDOW = 60          # ~10 minutes at 10s sampling
SPLIT_DATE = '2020-07-01'    # train on Failures 1-3, hold out Failure 4 for testing
H1_LOW_THRESHOLD = 1.0       # threshold for counting H1 "low" states (cycle-off detection)


def load_data(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def label_failures(df):
    df['label'] = 0
    for start, end, name in FAILURES:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        df.loc[mask, 'label'] = 1
    print("Label distribution:\n", df['label'].value_counts())
    return df


def engineer_features(df):
    print("Engineering features (this can take a few minutes on the full dataset)...")

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

    print("Feature engineering complete.")
    return df


def split_train_test(df, split_date=SPLIT_DATE):
    train = df[df['timestamp'] < split_date].copy()
    test = df[df['timestamp'] >= split_date].copy()
    print(f"Train: {train.shape} | failures: {train['label'].sum()}")
    print(f"Test:  {test.shape} | failures: {test['label'].sum()}")
    return train, test


def train_model(train, feature_cols):
    X_train = train[train['label'] == 0][feature_cols].dropna()
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    model.fit(X_train)
    print(f"Model trained on {X_train.shape[0]:,} normal rows")
    return model, X_train


def build_normal_stats(X_train, feature_cols):
    stats = X_train[feature_cols].agg(['mean', 'std']).T
    stats.columns = ['normal_mean', 'normal_std']
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/MetroPT3(AirCompressor).csv')
    parser.add_argument('--out', default='models/compressor_pipeline_v3.pkl')
    parser.add_argument('--threshold-percentile', type=float, default=5.0)
    parser.add_argument('--persistence-ratio', type=float, default=0.5)
    args = parser.parse_args()

    df = load_data(args.data)
    df = label_failures(df)
    df = engineer_features(df)

    train, test = split_train_test(df)
    model, X_train = train_model(train, FEATURE_COLS)
    normal_stats = build_normal_stats(X_train, FEATURE_COLS)

    pipeline = CompressorAnomalyPipeline(
        model=model,
        normal_stats=normal_stats,
        feature_cols=FEATURE_COLS,
        threshold_percentile=args.threshold_percentile,
        persistence_window=ROLLING_WINDOW,
        persistence_ratio=args.persistence_ratio,
    )

    X_test = test[FEATURE_COLS].dropna()
    test_scores = model.decision_function(X_test)
    pipeline.set_threshold(test_scores)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"Saved trained pipeline to {args.out}")

    # Quick sanity check on Failure 4 (held-out)
    fail4 = test[(test['timestamp'] >= '2020-07-15 14:30') & (test['timestamp'] <= '2020-07-15 19:00')]
    alerts = pipeline.run(fail4)
    print("\nFailure 4 (held-out) alert severity distribution:")
    print(alerts['severity'].value_counts())


if __name__ == "__main__":
    main()
