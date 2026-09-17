"""
app.py — unified NAUD backend, multi-well (Postgres-backed).

Simulates a small fleet of wells/compressors, each independently replaying the real
historical dataset (stored in Postgres, at a randomized starting offset, with small
cosmetic jitter for visual distinctness), each running the real trained pipeline on
its own buffer. Most wells loop through normal history indefinitely; any well can be
pointed at a known failure window on command, from a separate /control surface (not
the main dashboard), so a second device can drive the demo.

Run:
    cd backend
    uvicorn app:app --reload --port 8000

Dashboard: http://localhost:8000/         (what judges see)
Control:   http://localhost:8000/control  (what you run the demo from, on another device)

Data source: Postgres, via DATABASE_URL env var (Heroku sets this automatically once
the heroku-postgresql addon is attached). Table: readings, loaded via load_csv.py.
Model: MODEL_PATH env var, defaults to models/compressor_pipeline_v3.pkl.
"""

import os
import sys
import asyncio
import threading
from contextlib import asynccontextmanager

import numpy as np
import joblib
import pandas as pd
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

from pipeline import CompressorAnomalyPipeline  # noqa: E402
from explain import explain_alerts_batch  # noqa: E402
from replay import ReplaySimulator  # noqa: E402
from generate_demo_data import SENSORS  # noqa: E402

REAL_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, '..', 'models', 'compressor_pipeline_v3.pkl'))
READINGS_TABLE = os.environ.get("READINGS_TABLE", "readings")

_raw_db_url = os.environ.get("DATABASE_URL")
DATABASE_URL = _raw_db_url.replace("postgres://", "postgresql+psycopg2://", 1) if _raw_db_url else None

KNOWN_FAILURES = [
    ('2020-04-18 00:00', '2020-04-18 23:59', 'Failure 1 -- Air Leak'),
    ('2020-05-29 23:30', '2020-05-30 06:00', 'Failure 2 -- Air Leak'),
    ('2020-06-05 10:00', '2020-06-07 14:30', 'Failure 3 -- Air Leak'),
    ('2020-07-15 14:30', '2020-07-15 19:00', 'Failure 4 -- Air Leak (held-out test)'),
]

# Fleet layout. x/y are 0-100 abstract site-map coordinates, not real geography --
# purely for a small visual map so the fleet feels spatial, matching the "Site Map"
# concept in the team's original design.
WELL_DEFS = [
    {"id": "seg-2", "name": "Segment 2", "x": 22, "y": 28},
    {"id": "seg-4", "name": "Segment 4", "x": 58, "y": 62},
    {"id": "seg-7", "name": "Segment 7", "x": 78, "y": 22},
    {"id": "wellhead-7", "name": "Wellhead 7", "x": 34, "y": 78},
    {"id": "compressor-alpha", "name": "Compressor Alpha", "x": 64, "y": 40},
]
JITTER_FRAC = 0.03  # cosmetic display-only noise, see replay.py

state = {
    "pipeline": None,
    "engine": None,
    "using_demo": False,
    "failures": KNOWN_FAILURES,
    "wells": {},          # well_id -> {"name", "x", "y", "simulator"}
    "alerts": [],         # global feed, newest first, each tagged with well_id/well_name
    "next_alert_id": 1,
    "max_alerts": 500,
}


def load_pipeline_and_data():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Attach a Postgres addon (heroku addons:create "
            "heroku-postgresql:essential-0) or set DATABASE_URL locally."
        )
    print(f"Connecting to Postgres and loading model from {REAL_MODEL_PATH}")
    engine = create_engine(DATABASE_URL)
    pipeline = joblib.load(REAL_MODEL_PATH)
    state["using_demo"] = False
    state["failures"] = KNOWN_FAILURES
    return engine, pipeline


def make_on_tick(well_id, well_name):
    def on_tick(new_rows, window_df):
        pipeline = state["pipeline"]
        if pipeline is None or len(window_df) < 5:
            return
        try:
            alerts_df = pipeline.run(window_df, top_n=3)
        except Exception as e:
            print(f"[on_tick:{well_id}] pipeline error: {e}")
            return
        if len(alerts_df) == 0:
            return
        alerts_df = explain_alerts_batch(alerts_df)
        last = alerts_df.iloc[-1]
        top_feats = [
            {"feature": f[0], "z_score": round(float(f[1]), 3),
             "actual_value": round(float(f[2]), 3) if len(f) > 2 else None,
             "normal_mean": round(float(f[3]), 3) if len(f) > 3 else None}
            for f in last['top_features']
        ]
        well = state["wells"].get(well_id, {})
        record = {
            "id": state["next_alert_id"],
            "well_id": well_id,
            "well_name": well_name,
            "well_x": well.get("x"),
            "well_y": well.get("y"),
            "timestamp": str(last['timestamp']),
            "severity": last['severity'],
            "plain_language": last['plain_language'],
            "flag_rate": round(float(last['flag_rate']), 3),
            "top_features": top_feats,
        }
        existing = [a for a in state["alerts"] if a["well_id"] == well_id]
        if existing and existing[0]["timestamp"] == record["timestamp"]:
            return
        state["next_alert_id"] += 1
        state["alerts"].insert(0, record)
        state["alerts"] = state["alerts"][:state["max_alerts"]]
    return on_tick


def build_wells(engine, pipeline):
    rng = np.random.default_rng(7)
    bounds = pd.read_sql(f"SELECT min(ts) AS lo, max(ts) AS hi FROM {READINGS_TABLE}", engine).iloc[0]
    lo, hi = bounds["lo"], bounds["hi"]
    span_seconds = max(1, int((hi - lo).total_seconds()))
    wells = {}
    for i, wd in enumerate(WELL_DEFS):
        offset = int(rng.integers(0, span_seconds))
        start_ts = lo + pd.Timedelta(seconds=offset)
        sim = ReplaySimulator(
            engine, table=READINGS_TABLE, window_size=pipeline.persistence_window,
            tick_seconds=1.2, rows_per_tick=6,
            jitter_frac=JITTER_FRAC, sensor_cols=SENSORS,
            start_ts=start_ts, rng_seed=1000 + i,
        )
        wells[wd["id"]] = {"name": wd["name"], "x": wd["x"], "y": wd["y"], "simulator": sim}
    return wells


def start_background_replay():
    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state["loop"] = loop
        for well_id, well in state["wells"].items():
            loop.create_task(well["simulator"].run_forever(make_on_tick(well_id, well["name"])))
        loop.run_forever()
    t = threading.Thread(target=runner, daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, pipeline = load_pipeline_and_data()
    state["pipeline"] = pipeline
    state["engine"] = engine
    state["wells"] = build_wells(engine, pipeline)
    start_background_replay()
    yield
    for well in state["wells"].values():
        well["simulator"].stop()


app = FastAPI(title="NAUD Compressor Monitoring Backend", version="2.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class JumpRequest(BaseModel):
    well_id: str
    failure_id: int


class SpeedRequest(BaseModel):
    well_id: Optional[str] = None  # omit to apply to all wells
    tick_seconds: Optional[float] = None
    rows_per_tick: Optional[int] = None


class PlayPauseRequest(BaseModel):
    well_id: Optional[str] = None  # omit to apply to all wells


def _well_status(well_id):
    for a in state["alerts"]:
        if a["well_id"] == well_id:
            return a["severity"]
    return "NOMINAL"


@app.get("/health")
def health():
    wells_summary = {
        wid: {"running": w["simulator"].running, "cursor_ts": str(w["simulator"].cursor_ts)}
        for wid, w in state["wells"].items()
    }
    total_rows = None
    if state.get("engine") is not None:
        total_rows = int(pd.read_sql(f"SELECT count(*) AS n FROM {READINGS_TABLE}", state["engine"])["n"][0])
    return {
        "status": "ok",
        "using_demo_data": state["using_demo"],
        "model_loaded": state["pipeline"] is not None,
        "total_rows": total_rows,
        "well_count": len(state["wells"]),
        "wells": wells_summary,
    }


@app.get("/wells")
def list_wells():
    out = []
    for wid, w in state["wells"].items():
        sim = w["simulator"]
        reading = sim.latest_reading()
        out.append({
            "id": wid,
            "name": w["name"],
            "x": w["x"],
            "y": w["y"],
            "status": _well_status(wid),
            "running": sim.running,
            "reading": {k: (str(v) if k == 'timestamp' else v)
                        for k, v in reading.items()} if reading else None,
        })
    return {"wells": out}


@app.get("/wells/{well_id}/live-reading")
def well_live_reading(well_id: str):
    well = state["wells"].get(well_id)
    if well is None:
        raise HTTPException(status_code=404, detail=f"Unknown well_id: {well_id}")
    reading = well["simulator"].latest_reading()
    if reading is None:
        return {"reading": None}
    reading = {k: (str(v) if k == 'timestamp' else v) for k, v in reading.items()}
    return {"reading": reading}


@app.get("/alerts")
def alerts(limit: int = 50, well_id: Optional[str] = None):
    items = state["alerts"]
    if well_id:
        items = [a for a in items if a["well_id"] == well_id]
    return {"alert_count": len(items), "alerts": items[:limit]}


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: int):
    for a in state["alerts"]:
        if a["id"] == alert_id:
            return a
    raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/failures")
def failures():
    return {"failures": [{"start": s, "end": e, "name": n} for s, e, n in state["failures"]]}


@app.post("/control/jump")
def jump(req: JumpRequest):
    well = state["wells"].get(req.well_id)
    if well is None:
        raise HTTPException(status_code=404, detail=f"Unknown well_id: {req.well_id}")
    try:
        well["simulator"].jump_to_failure(state["failures"], req.failure_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    state["alerts"] = [a for a in state["alerts"] if a["well_id"] != req.well_id]
    return {"status": "jumped", "well_id": req.well_id, "cursor_ts": str(well["simulator"].cursor_ts)}


@app.post("/control/speed")
def speed(req: SpeedRequest = SpeedRequest()):
    targets = [state["wells"][req.well_id]] if req.well_id else list(state["wells"].values())
    if req.well_id and req.well_id not in state["wells"]:
        raise HTTPException(status_code=404, detail=f"Unknown well_id: {req.well_id}")
    for w in targets:
        if req.tick_seconds is not None:
            w["simulator"].tick_seconds = req.tick_seconds
        if req.rows_per_tick is not None:
            w["simulator"].rows_per_tick = req.rows_per_tick
    return {"status": "updated", "applied_to": req.well_id or "all wells"}


@app.post("/control/play")
def play(req: PlayPauseRequest = PlayPauseRequest()):
    targets = [(req.well_id, state["wells"].get(req.well_id))] if req.well_id else list(state["wells"].items())
    for wid, w in targets:
        if w and not w["simulator"].running:
            loop = state.get("loop")
            if loop:
                loop.create_task(w["simulator"].run_forever(make_on_tick(wid, w["name"])))
    return {"status": "playing", "applied_to": req.well_id or "all wells"}


@app.post("/control/pause")
def pause(req: PlayPauseRequest = PlayPauseRequest()):
    targets = [state["wells"][req.well_id]] if req.well_id else list(state["wells"].values())
    if req.well_id and req.well_id not in state["wells"]:
        raise HTTPException(status_code=404, detail=f"Unknown well_id: {req.well_id}")
    for w in targets:
        w["simulator"].running = False
    return {"status": "paused", "applied_to": req.well_id or "all wells"}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(HERE, "dashboard.html"), "r") as f:
        return f.read()


@app.get("/control", response_class=HTMLResponse)
def control_page():
    with open(os.path.join(HERE, "control.html"), "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
