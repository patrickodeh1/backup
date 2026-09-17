"""
replay.py — ReplaySimulator

Replays historical sensor data in timestamp order, on a timer, as a stand-in for a
live SCADA feed. This is deliberately NOT a synthetic signal generator: it plays
back real recorded rows (or the faithful demo dataset), so everything downstream
sees genuine sensor texture instead of hand-tuned noise.

Designed for a live demo: you can jump the playhead to a specific point (e.g. the
start of a known failure window) on command, so a presenter can control exactly
when the "live" failure appears without waiting for real-time playback.
"""

import asyncio
import time
import numpy as np
import pandas as pd


class ReplaySimulator:
    def __init__(self, df, window_size=60, tick_seconds=1.0, rows_per_tick=6,
                 jitter_frac=0.0, sensor_cols=None, start_cursor=None, rng_seed=None):
        """
        df: full historical dataframe, sorted by timestamp, with raw sensor columns.
        window_size: how many trailing rows to keep in the live buffer (must be >=
            the pipeline's rolling window, default 60, for features to be meaningful).
        tick_seconds: wall-clock seconds between ticks.
        rows_per_tick: how many historical rows to advance per tick. Real sampling
            is one row per 10s; at rows_per_tick=6 and tick_seconds=1.0, one hour of
            real time plays back in 100 seconds -- fast enough for a live demo,
            slow enough to watch alerts appear.
        jitter_frac: if > 0, adds small gaussian noise to sensor_cols on each replayed
            row, sized as jitter_frac * that sensor's overall std. This is perturbation
            of real recorded values, not synthetic generation -- it exists so multiple
            simulated wells replaying the same history don't show identical numbers,
            and so numbers aren't perfectly predictable/repeatable run to run.
        sensor_cols: raw sensor columns to jitter. Required if jitter_frac > 0.
        start_cursor: start the playhead here instead of row 0 (e.g. a randomized
            offset per well, so a fleet of wells isn't all in lockstep).
        rng_seed: seed for this instance's jitter RNG, for reproducible-but-distinct wells.
        """
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.tick_seconds = tick_seconds
        self.rows_per_tick = rows_per_tick
        self.jitter_frac = jitter_frac
        self.sensor_cols = sensor_cols or []
        self.rng = np.random.default_rng(rng_seed)
        self._sensor_std = {c: self.df[c].std() for c in self.sensor_cols} if jitter_frac else {}
        self.cursor = start_cursor if start_cursor is not None else 0
        self.running = False
        self.buffer = pd.DataFrame(columns=df.columns)
        self._lock = asyncio.Lock()

    def jump_to(self, index_or_timestamp):
        """Move the playhead. Accepts a row index, or a timestamp string/Timestamp."""
        if isinstance(index_or_timestamp, (int,)):
            idx = index_or_timestamp
        else:
            ts = pd.to_datetime(index_or_timestamp)
            matches = self.df.index[self.df['timestamp'] >= ts]
            idx = int(matches[0]) if len(matches) else len(self.df) - 1
        # back up enough rows so the rolling window fills with real context, not a cold start
        idx = max(0, idx - self.window_size)
        self.cursor = idx

    def jump_to_failure(self, failures, failure_id):
        """failures: list of (start, end, name) tuples. failure_id: 1-indexed."""
        if failure_id < 1 or failure_id > len(failures):
            raise ValueError(f"failure_id must be 1..{len(failures)}")
        start, _, _ = failures[failure_id - 1]
        self.jump_to(start)

    def reset(self):
        self.cursor = 0
        self.buffer = pd.DataFrame(columns=self.df.columns)

    def step(self):
        """Advance the playhead by rows_per_tick rows, update the rolling buffer."""
        if self.cursor >= len(self.df):
            self.cursor = 0  # loop back to start for a continuous demo
        end = min(self.cursor + self.rows_per_tick, len(self.df))
        new_rows = self.df.iloc[self.cursor:end].copy()
        self.cursor = end

        # NOTE: jitter is intentionally NOT applied here. The buffer below feeds the
        # real trained pipeline -- detection must run on genuine recorded values, so the
        # ML layer stays the untouched source of truth. Jitter (for visual per-well
        # distinctness) is applied only in latest_reading(), which is display-only.
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
