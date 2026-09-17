"""
api.py — FastAPI service for the compressor anomaly detection pipeline.

Wraps the trained CompressorAnomalyPipeline + plain-language layer behind a simple
HTTP API, so the NAUD dashboard/backend can call this without needing Python,
the model file, or any ML dependencies locally.

Run locally:
    uvicorn src.api:app --reload --port 8000

Endpoints:
    GET  /health              -> basic liveness check
    POST /analyze              -> takes raw sensor rows, returns alerts

Example request body for POST /analyze:
{
  "rows": [
    {"timestamp": "2020-07-15T14:30:00", "TP2": 8.2, "TP3": 8.9, "H1": 0.1,
     "DV_pressure": 0.02, "Reservoirs": 8.9, "Oil_temperature": 85.2, "Motor_current": 5.6},
    ... (needs enough consecutive rows to fill the rolling window, ~60 rows / 10 min)
  ]
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from pipeline import CompressorAnomalyPipeline  # noqa: F401 -- required for unpickling
from explain import explain_alerts_batch
from train import engineer_features, SENSORS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/compressor_pipeline_v3.pkl")

app = FastAPI(title="NAUD Compressor Anomaly Detection API", version="1.0")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


class SensorRow(BaseModel):
    timestamp: str
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float


class AnalyzeRequest(BaseModel):
    rows: List[SensorRow]
    top_n_features: Optional[int] = 3


class Alert(BaseModel):
    timestamp: str
    severity: str
    plain_language: str
    flag_rate: float
    top_features: List[dict]


class AnalyzeResponse(BaseModel):
    alert_count: int
    alerts: List[Alert]


@app.get("/health")
def health():
    try:
        p = get_pipeline()
        return {"status": "ok", "model_loaded": True, "feature_count": len(p.feature_cols)}
    except Exception as e:
        return {"status": "error", "model_loaded": False, "detail": str(e)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    pipeline = get_pipeline()

    if len(request.rows) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need multiple consecutive sensor rows (ideally ~60, covering "
                   "~10 minutes) for rolling-window features to be meaningful."
        )

    df = pd.DataFrame([r.dict() for r in request.rows])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    missing = [s for s in SENSORS if s not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required sensor columns: {missing}")

    try:
        df = engineer_features(df)
        alerts_df = pipeline.run(df, top_n=request.top_n_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if len(alerts_df) == 0:
        return AnalyzeResponse(alert_count=0, alerts=[])

    alerts_df = explain_alerts_batch(alerts_df)

    alerts_out = []
    for _, row in alerts_df.iterrows():
        top_feats_serializable = [
            {"feature": f[0], "z_score": round(f[1], 3),
             "actual_value": round(f[2], 3) if len(f) > 2 else None,
             "normal_mean": round(f[3], 3) if len(f) > 3 else None}
            for f in row['top_features']
        ]
        alerts_out.append(Alert(
            timestamp=str(row['timestamp']),
            severity=row['severity'],
            plain_language=row['plain_language'],
            flag_rate=round(float(row['flag_rate']), 3),
            top_features=top_feats_serializable,
        ))

    return AnalyzeResponse(alert_count=len(alerts_out), alerts=alerts_out)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
