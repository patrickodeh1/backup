"""
replay.py — ReplaySimulator (Postgres-backed)

Replays historical sensor data in timestamp order, on a timer, as a stand-in for a
live SCADA feed, reading from a Postgres table instead of an in-memory dataframe.
Each tick pulls only the next few rows via a keyset (timestamp) query, and the
rolling feature window is only ever ~window_size*3 rows in memory -- the full
history never has to be loaded or feature-engineered at once.
"""

import asyncio
import numpy as np
import pandas as pd


class ReplaySimulator:
    def __init__(self, engine, table="readings", window_size=60, tick_seconds=1.0,
                 rows_per_tick=6, jitter_frac=0.0, sensor_cols=None,
                 start_ts=None, rng_seed=None):
        self.engine = engine
        self.table = table
        self.window_size = window_size
        self.tick_seconds = tick_seconds
        self.rows_per_tick = rows_per_tick
        self.jitter_frac = jitter_frac
        self.sensor_cols = sensor_cols or []
        self.rng = np.random.default_rng(rng_seed)
        self._sensor_std = self._load_stds() if jitter_frac else {}
        self.min_ts, self.max_ts = self._bounds()
        self.cursor_ts = start_ts if start_ts is not None else self.min_ts
        self.running = False
        self.buffer = pd.DataFrame()
        self._lock = asyncio.Lock()

    def _bounds(self):
        row = pd.read_sql(f"SELECT min(ts) AS lo, max(ts) AS hi FROM {self.table}", self.engine).iloc[0]
        return row["lo"], row["hi"]

    def _load_stds(self):
        cols = ", ".join(f"stddev({c}) AS {c}" for c in self.sensor_cols)
        row = pd.read_sql(f"SELECT {cols} FROM {self.table}", self.engine).iloc[0]
        return row.to_dict()

    def jump_to(self, timestamp):
        """Move the playhead near a given timestamp, backed up window_size rows
        so the rolling window fills with real context, not a cold start."""
        ts = pd.to_datetime(timestamp)
        q = f"""SELECT ts FROM {self.table} WHERE ts <= %(ts)s
                ORDER BY ts DESC LIMIT 1 OFFSET {self.window_size}"""
        row = pd.read_sql(q, self.engine, params={"ts": ts})
        self.cursor_ts = row["ts"].iloc[0] if len(row) else self.min_ts
        self.buffer = pd.DataFrame()

    def jump_to_failure(self, failures, failure_id):
        """failures: list of (start, end, name) tuples. failure_id: 1-indexed."""
        if failure_id < 1 or failure_id > len(failures):
            raise ValueError(f"failure_id must be 1..{len(failures)}")
        start, _, _ = failures[failure_id - 1]
        self.jump_to(start)

    def reset(self):
        self.cursor_ts = self.min_ts
        self.buffer = pd.DataFrame()

    def step(self):
        """Advance the playhead by rows_per_tick rows, update the rolling buffer."""
        q = f"""SELECT * FROM {self.table} WHERE ts >= %(cur)s
                ORDER BY ts LIMIT {self.rows_per_tick}"""
        new_rows = pd.read_sql(q, self.engine, params={"cur": self.cursor_ts})
        if len(new_rows) == 0:
            self.cursor_ts = self.min_ts  # loop back to start for a continuous demo
            return pd.DataFrame()
        self.cursor_ts = new_rows["ts"].iloc[-1] + pd.Timedelta(seconds=1)
        # Postgres lowercases all column names. The pipeline / engineer_features /
        # dashboard all expect the original CSV casing throughout -- rename every
        # column back here, once, right after reading from the DB, so nothing
        # downstream (buffer, current_window, pipeline.run, engineer_features)
        # ever has to know Postgres was involved.
        new_rows = new_rows.rename(columns={
            "ts": "timestamp",
            "tp2": "TP2", "tp3": "TP3", "h1": "H1",
            "dv_pressure": "DV_pressure", "reservoirs": "Reservoirs",
            "oil_temperature": "Oil_temperature", "motor_current": "Motor_current",
            "comp": "COMP", "dv_eletric": "DV_eletric", "towers": "Towers",
            "mpg": "MPG", "lps": "LPS", "pressure_switch": "Pressure_switch",
            "oil_level": "Oil_level", "caudal_impulses": "Caudal_impulses",
        })
        self.buffer = pd.concat([self.buffer, new_rows]).tail(self.window_size * 3)
        return new_rows

    def current_window(self):
        """Return the trailing window_size rows for feature computation."""
        return self.buffer.tail(self.window_size).copy()

    def latest_reading(self):
        """Display-only snapshot. Jitter (if configured) is applied here, never to
        the buffer the pipeline reads, so detection always runs on real recorded data."""
        if len(self.buffer) == 0:
            return None
        reading = self.buffer.iloc[-1].to_dict()
        if self.jitter_frac and self.sensor_cols:
            for c in self.sensor_cols:
                std = self._sensor_std.get(c, 0) or 0
                if std > 0 and c in reading and reading[c] is not None:
                    reading[c] = float(reading[c]) + float(self.rng.normal(0, std * self.jitter_frac))
        return reading

    async def run_forever(self, on_tick):
        """
        on_tick(new_rows, window_df) is called each tick with the newly replayed
        rows and the current trailing window. Runs until self.running is set False.
        """
        self.running = True
        while self.running:
            async with self._lock:
                new_rows = self.step()
                window_df = self.current_window()
            try:
                on_tick(new_rows, window_df)
            except Exception as e:
                print(f"[replay] on_tick error: {e}")
            await asyncio.sleep(self.tick_seconds)

    def stop(self):
        self.running = False
