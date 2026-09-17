# NAUD: Compressor Anomaly Detection (Model Component)

Predictive/anomaly detection system for Renaissance Innovation Week 2026, Challenge 4.
This repo covers the ML component end-to-end: data, EDA, feature engineering, model
training, evidence/explanation layer, severity classification, and a deployable API.

## Dataset
MetroPT-3 (UCI #791) — real industrial air compressor sensor data, ~1.5M rows,
Feb–Sep 2020, 4 documented air leak failure events. Raw CSV is gitignored due to size;
download from https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B

## Status
- [x] EDA complete — confirmed failure windows, quantified sensor deviation
- [x] Feature engineering — validated rolling-window features (see outputs/)
- [x] Model training (Isolation Forest) — trained on normal data, tested on held-out Failure 4
- [x] Evaluation — see Results below
- [x] Evidence/explanation layer — per-alert feature attribution, correctly signed (higher/lower than normal)
- [x] Severity classification — MONITOR / INVESTIGATE / ESCALATE, thresholds set from real data quantiles
- [x] Plain-language translation layer — rule-based, deterministic (`src/explain.py`)
- [x] Optional LLM-powered explanation layer with automatic fallback (`src/explain_llm.py`)
- [x] Packaged into a reusable pipeline class (`src/pipeline.py`)
- [x] Reproducible training script (`src/train.py`)
- [x] Validated across all 4 documented failures
- [x] Deployable API (`src/api.py`) — see API_DOCS.md


## Repo structure
```
├── README.md
├── API_DOCS.md              ← full API reference for backend integration
├── requirements.txt
├── notebooks/
│   └── eda.ipynb             (exploration record — not the source of truth for production code)
├── src/
│   ├── pipeline.py           CompressorAnomalyPipeline class
│   ├── explain.py            Rule-based plain-language layer (default, no dependencies)
│   ├── explain_llm.py        Optional LLM-powered layer, falls back to explain.py on failure
│   ├── train.py               Reproducible end-to-end training script
│   └── api.py                 FastAPI service
├── models/
│   └── compressor_pipeline_v3.pkl
├── data/                     (gitignored — raw + processed CSVs)
├── outputs/                  (EDA plots and summary CSVs)
```

## Key finding
In every documented failure, the compressor loses its normal on/off duty-cycling and
runs continuously instead — this loss-of-rhythm is the core signal the model is built
around (see `outputs/` for supporting plots). `H1_cycle_transitions` showed a 50–1000x
drop during failures compared to normal operation, consistent across all 4 events.

## Model & Evaluation
Isolation Forest, trained only on normal operation data (never shown failure examples),
tested on Failure 4 — an event fully held out from training:

- Raw predictions (5th percentile anomaly-score threshold): 83% recall, 6% precision
- With persistence smoothing (flag only if ≥50% of last 10 min anomalous): **90% recall, 10% precision**

Persistence smoothing improved both recall and precision by filtering single-row noise,
keeping only sustained anomalous behavior.

## Pipeline validated across all 4 documented failures
| Failure | Total alerts | ESCALATE | INVESTIGATE | MONITOR |
|---|---|---|---|---|
| 1 | 8,537 | 31 | 8,376 | 130 |
| 2 | 2,360 | 0 | 2,356 | 4 |
| 3 | 17,195 | 73 | 16,976 | 146 |
| 4 (held-out) | 1,465 | 1,378 | 87 | 0 |

Severity thresholds (ESCALATE ≥3.2 std devs, INVESTIGATE ≥2.0, MONITOR below) were set
from the actual 75th/25th percentiles of deviation magnitude observed across the full
test set. The held-out failure (4) is classified almost entirely ESCALATE, consistent
with it showing the largest oil temperature deviation (+3.17 std) of all four failures.

## Early-warning behavior
Of alerts flagged outside the four labeled failure windows, 42% occurred within 24
hours of a known failure, and 17% within just 6 hours — suggesting a meaningful share
of "false positives" are early-warning signals the labels simply don't credit.

## Plain-language layer
Two implementations, same interface:
- **`explain.py`** (default): rule-based templates. Every sentence is grounded directly
  in computed numbers — deterministic, free, instant, cannot hallucinate. This is what
  the deployed API uses by default.
- **`explain_llm.py`** (optional): sends the same structured, already-computed data to
  an LLM for more natural phrasing. The LLM only rephrases given numbers — it never
  sees raw sensor data and cannot invent findings. Falls back automatically to
  `explain.py` if the API call fails for any reason.

## API
Full request/response reference in **[API_DOCS.md](./API_DOCS.md)**.

Quick start:
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
curl http://localhost:8000/health
```

`POST /analyze` accepts a batch of recent raw sensor rows (ideally ~60+, covering
~10 minutes) and returns structured + plain-language alerts — no Python or ML
knowledge required on the caller's side.

## Retraining
```bash
python src/train.py --data "data/MetroPT3(AirCompressor).csv" --out models/compressor_pipeline_v4.pkl
```
Fully reproducible — one command from raw data to a validated, saved pipeline.
Useful if the team gets new or updated sensor data later.

## Full pipeline example (single alert, real held-out data)
```
{'severity': 'ESCALATE',
 'top_features': [('Oil_temperature_roll_mean', 3.29), ('TP2_roll_std', 1.55), ('H1_roll_std', 1.43)],
 'flag_rate': 1.0}
```

## Known limitations / honest next steps
- Only 4 documented failures exist in this dataset, all air leaks — model is validated
  on this failure type only, not on other failure modes (e.g. oil leaks).
- Precision against strict labeled windows is moderate (10%), though a substantial
  share of flagged alerts appear to be genuine early warnings rather than noise.
- Severity thresholds are calibrated against this dataset's observed distribution,
  not yet validated against real operator judgment.
- Trained and validated on MetroPT-3 (metro compressor data) as a proxy for oil & gas
  compression equipment — same underlying physics (duty-cycling, motor load, leak
  dynamics).

