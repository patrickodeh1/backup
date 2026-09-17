# Running the unified backend + dashboard

## 1. Drop these files into your repo

Copy this whole `backend/` folder into your `nuad-ai` repo root (alongside your existing `src/`, `data/`, `models/`).

## 2. Install dependencies (if not already)

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

## 3. Point it at your real data and model

The backend looks for these paths by default:

- `data/MetroPT3(AirCompressor).csv`
- `models/compressor_pipeline_v3.pkl`

If your real files are already at those paths relative to the repo root, you don't need to do anything else. If they're somewhere else, set env vars before running:

```bash
export DATA_PATH=/path/to/MetroPT3.csv
export MODEL_PATH=/path/to/compressor_pipeline_v3.pkl
```

If neither file is found, the backend automatically builds and uses a synthetic fallback dataset so you can still test everything works. You'll see `"using_demo_data": true` in `/health` and a "DEMO DATA" tag on the dashboard when this happens — swap in your real files before the actual pitch.

**Note on your real CSV:** it likely only has raw sensor columns, not the engineered rolling-window features your model expects. The backend now checks for this at startup and, if any feature column your model needs is missing, runs feature engineering once over the full history before starting the replay. On the full ~1.5M row dataset this can take a couple of minutes on first startup — that's expected, not a hang. Subsequent requests are fast once it's done.

## 4. Run it

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** — that's the dashboard, served directly by the backend.

## 5b. About the dashboard design

This dashboard is built fresh in HTML/JS (not a copy of the team's actual React code — I didn't have repo access), but it matches their real design language: same NAUD wordmark and orange accent, same dark sidebar, same stat-card and gauge style. It's been simplified from their version: no Manual-entry page (the model doesn't consume it, and it added navigation with no purpose here), no profile icon (no auth in this build). The History page's Anomaly Alert detail view is modeled closely on their own screenshot — AI Analysis, Recommended Action, Associated Telemetry, a small site map — but every field is real, driven by actual z-scores and actual explanations, not placeholder text.

If the real repo becomes reachable before the pitch, the right move is to port this same data-fetching logic into their actual components, so their original code ships, just wired to a live model instead of hardcoded numbers.

## 5c. Multi-well simulation

Rather than replaying one fixed CSV as a single static feed, the backend now runs a small fleet of 5 simulated wells (Segment 2, Segment 4, Segment 7, Wellhead 7, Compressor Alpha — names picked to match the team's original mockup). Each well:

- Has its own independent cursor into the real historical data, starting at a randomized offset, so no two wells show the same numbers at the same time
- Has small cosmetic jitter (±3% of each sensor's real standard deviation) added only to the *displayed* readings — this exists purely so the fleet doesn't look mechanically identical, and it never touches the data the model actually runs on. Detection always runs on genuine recorded values, unmodified
- Loops through normal history indefinitely by default, so most of the fleet just sits there looking calm and real
- Can be pointed at any known failure window on command, independently, from the `/control` page

This means at any moment the fleet looks genuinely alive — small unpredictable movements everywhere, most wells boring — and you can turn any one of them into a live incident on cue.

## 5d. The /control page

Demo controls (play/pause, speed, jump-to-failure) live at `http://localhost:8000/control`, not on the main dashboard. Open that on a second device (your phone, a second laptop) so you can drive the demo without touching the screen judges are looking at. The main dashboard (`/`) has no controls on it at all now.

## 6. What's running

- Five simulated wells, each independently replaying your historical CSV in timestamp order, on a timer, as a stand-in live feed
- Every tick, per well, the new window of rows runs through your real `CompressorAnomalyPipeline` and `explain.py`
- The dashboard polls `/wells` and `/alerts` and updates in real time
- The `/control` page lets you play/pause/speed up/jump-to-failure each well independently, or all at once

## 7. For the pitch

- Open `/control` on a second device before you start
- Use the speed controls to fast-forward a well through quiet normal periods, then land on 1x right before triggering a failure, so judges watch it happen in real time rather than waiting
- The "Jump to failure" buttons per well are your safety net — you control exactly when and where the failure appears, no waiting on luck. Trigger it on one well while the other four stay calm in the background, which is a stronger visual than everything alerting at once
- If you want a clean recorded demo video as backup in case of live technical issues, screen-record one full run through: fleet calm → jump one well to a failure → watch ESCALATE fire → click into the alert detail

## Endpoints reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness, which dataset is loaded, per-well running state |
| `/wells` | GET | all wells with status, location, current reading |
| `/wells/{well_id}/live-reading` | GET | one well's latest raw sensor snapshot |
| `/alerts?limit=50&well_id=...` | GET | recent alerts across the fleet (or one well), newest first |
| `/alerts/{alert_id}` | GET | single alert detail |
| `/failures` | GET | list of known failure windows |
| `/control/jump` | POST | `{"well_id": "seg-4", "failure_id": 1}` |
| `/control/speed` | POST | `{"well_id": "seg-4", "tick_seconds": 0.1, "rows_per_tick": 6}` — omit `well_id` to apply to all |
| `/control/play` / `/control/pause` | POST | `{"well_id": "seg-4"}` — omit to apply to all |

## If your dashboard team wants to build their own frontend instead

They can ignore `dashboard.html` and `control.html` entirely and just poll `/wells` and `/alerts` from their own frontend against this same backend — the data contract doesn't change either way.
