"""
pipeline.py — CompressorAnomalyPipeline

Full detect -> explain -> classify severity pipeline for compressor anomaly detection.
Trained model expects engineered rolling-window features (see notebooks/eda.ipynb for
how these are built from raw MetroPT-3-style sensor data).
"""

import numpy as np
import pandas as pd


class CompressorAnomalyPipeline:
    def __init__(self, model, normal_stats, feature_cols,
                 threshold_percentile=5, persistence_window=60, persistence_ratio=0.5):
        self.model = model
        self.normal_stats = normal_stats  # DataFrame indexed by feature, columns: normal_mean, normal_std
        self.feature_cols = feature_cols
        self.threshold_percentile = threshold_percentile
        self.persistence_window = persistence_window
        self.persistence_ratio = persistence_ratio
        self.threshold_ = None

    def set_threshold(self, reference_scores):
        """Call once on a reference/validation set to lock in the anomaly threshold."""
        self.threshold_ = np.percentile(reference_scores, self.threshold_percentile)

    def score(self, X):
        """Raw anomaly scores -- lower = more anomalous."""
        return self.model.decision_function(X[self.feature_cols])

    def detect(self, df_chunk):
        """
        Run detection on a chunk of data with a 'timestamp' column and all
        feature_cols present. Returns df_chunk with added columns:
        raw_flag, flag_rate, smoothed_flag.
        """
        df_chunk = df_chunk.sort_values('timestamp').reset_index(drop=True)
        scores = self.score(df_chunk)
        df_chunk['raw_flag'] = (scores < self.threshold_).astype(int)
        df_chunk['flag_rate'] = df_chunk['raw_flag'].rolling(
            self.persistence_window, min_periods=1).mean()
        df_chunk['smoothed_flag'] = (df_chunk['flag_rate'] >= self.persistence_ratio).astype(int)
        return df_chunk

    def explain(self, row, top_n=3, with_direction=True):
        """
        Return the top_n most deviating features for this row.

        If with_direction=True (default), returns 4-tuples:
            (feature_name, z_score, actual_value, normal_mean)
        so downstream plain-language code can correctly say "higher" or "lower"
        than normal, instead of only "unusual".

        If with_direction=False, returns 2-tuples (feature_name, z_score) --
        kept for backward compatibility with earlier notebook code.
        """
        deviations = {}
        for feat in self.feature_cols:
            val = row[feat]
            mean = self.normal_stats.loc[feat, 'normal_mean']
            std = self.normal_stats.loc[feat, 'normal_std']
            if std > 0:
                z = abs((val - mean) / std)
                deviations[feat] = (z, val, mean)

        top_items = sorted(deviations.items(), key=lambda x: -x[1][0])[:top_n]

        if with_direction:
            return [(name, z, val, mean) for name, (z, val, mean) in top_items]
        else:
            return [(name, z) for name, (z, val, mean) in top_items]

    def classify_severity(self, top_features, flag_rate):
        """
        top_features: output of explain() -- either 2-tuples or 4-tuples, both work
        since we only read the z_score (index 1) here.
        """
        max_z = top_features[0][1]
        if max_z >= 3.2:
            return "ESCALATE"
        elif max_z >= 2.0:
            return "INVESTIGATE"
        else:
            return "MONITOR"

    def run(self, df_chunk, top_n=3):
        """Full pipeline: detect -> for each smoothed-flagged row, explain + classify."""
        detected = self.detect(df_chunk)
        alerts = []
        flagged_rows = detected[detected['smoothed_flag'] == 1]
        for idx, frow in flagged_rows.iterrows():
            top_feats = self.explain(frow, top_n=top_n, with_direction=True)
            severity = self.classify_severity(top_feats, frow['flag_rate'])
            alerts.append({
                'timestamp': frow['timestamp'],
                'severity': severity,
                'top_features': top_feats,
                'flag_rate': frow['flag_rate']
            })
        return pd.DataFrame(alerts)
