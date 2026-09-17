# Compressor Anomaly Detection API

A service that analyzes compressor sensor readings and returns anomaly alerts with
severity ratings and plain-language explanations.

---

## Overview

Send a batch of recent sensor readings, receive back a list of alerts (if any),
each with:
- A severity rating (`MONITOR`, `INVESTIGATE`, or `ESCALATE`)
- A plain-language explanation suitable for display to field staff
- The specific sensor readings that triggered the alert

---

## Base URL

| Environment | URL |
|---|---|
| Local development | `http://localhost:8000` |
| Production | *to be added* |

---

## Authentication

Not currently required. Recommend adding an API key or placing this service behind
a gateway before exposing the production URL externally.

---

## Endpoints

### `GET /health`

Returns service status and confirms the model is loaded.

**Example request**
```bash
curl https://<base-url>/health
```

**Example response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_count": 12
}
```

---

### `POST /analyze`

Analyzes a batch of sensor readings and returns any detected anomalies.

**Minimum batch size:** at least 60 consecutive readings (approximately 10 minutes
at 10-second sampling intervals) are recommended. The model relies on short-term
trends, not single readings, so smaller batches will still run but may be less
reliable.

#### Request body

```json
{
  "rows": [
    {
      "timestamp": "2020-07-15T14:30:00",
      "TP2": 8.2,
      "TP3": 8.9,
      "H1": 0.1,
      "DV_pressure": 0.02,
      "Reservoirs": 8.9,
      "Oil_temperature": 85.2,
      "Motor_current": 5.6
    }
  ],
  "top_n_features": 3
}
```

**Fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `rows` | array | Yes | Sensor readings. See schema below. Order is not required — sorted internally by timestamp. |
| `top_n_features` | integer | No (default: 3) | Number of contributing sensor readings to include per alert. |

**Sensor reading schema** (each object in `rows`)

| Field | Type | Description |
|---|---|---|
| `timestamp` | string (ISO 8601) | Reading timestamp |
| `TP2` | float | Primary pressure sensor |
| `TP3` | float | Secondary pressure sensor |
| `H1` | float | Compressor cycling sensor |
| `DV_pressure` | float | Differential valve pressure |
| `Reservoirs` | float | Reservoir pressure |
| `Oil_temperature` | float | Oil temperature |
| `Motor_current` | float | Motor current draw |

**Example request**
```bash
curl -X POST https://<base-url>/analyze \
  -H "Content-Type: application/json" \
  -d @payload.json
```

#### Response — `200 OK`

```json
{
  "alert_count": 1,
  "alerts": [
    {
      "timestamp": "2020-07-15 14:39:50",
      "severity": "ESCALATE",
      "plain_language": "URGENT: This compressor is showing strong signs of a developing problem. What we're seeing: oil temperature is running hotter than normal (3.3 standard deviations from normal). Recommended action: dispatch a maintenance check as soon as possible.",
      "flag_rate": 1.0,
      "top_features": [
        {
          "feature": "Oil_temperature_roll_mean",
          "z_score": 3.29,
          "actual_value": 83.9,
          "normal_mean": 62.6
        }
      ]
    }
  ]
}
```

If no anomalies are found, the response is `{"alert_count": 0, "alerts": []}`.
This is a normal, valid response — not an error.

**Response fields**

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | When the alert occurred |
| `severity` | string | `MONITOR`, `INVESTIGATE`, or `ESCALATE` |
| `plain_language` | string | Human-readable explanation and recommended action |
| `flag_rate` | float (0.5–1.0) | Proportion of the recent time window showing anomalous behavior |
| `top_features` | array | Sensor readings that most contributed to the alert, ranked by deviation |

#### Error responses

| Status | Cause |
|---|---|
| `400 Bad Request` | Fewer than 2 readings sent, or required sensor fields missing |
| `500 Internal Server Error` | Unexpected processing error |

---

## Severity levels

| Severity | Meaning | Suggested response |
|---|---|---|
| `MONITOR` | Mild, early-stage deviation | No immediate action — continue routine monitoring |
| `INVESTIGATE` | Moderate, sustained deviation | Schedule an inspection |
| `ESCALATE` | Strong, sustained deviation | Dispatch maintenance as soon as possible |

Severity thresholds are calibrated against observed deviation patterns in historical
compressor data, not fixed arbitrary values.

---

## Notes for integration

- Alerts are only returned for readings that show *sustained* anomalous behavior
  (at least 50% of the trailing 10-minute window), which filters out single-reading
  noise.
- The `plain_language` field is safe to display directly to end users — it is
  generated from a controlled template grounded in the underlying data, not
  freely generated text.
- For best results, submit overlapping or rolling batches (e.g. every few minutes,
  each containing the last 10+ minutes of readings) rather than isolated one-off
  batches, so the service always has enough recent context.
