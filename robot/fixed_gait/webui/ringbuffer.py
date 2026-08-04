"""Telemetry ring buffer: daemon writes ~20 Hz samples, HTTP readers copy slices by sequence."""
import threading

import numpy as np

import paths


class TelemetryRing:
    # cmd_* is the COMMANDED target (NaN when nothing is being commanded), kept in the same ring as
    # the measurement so the chart never has to align two independently-sampled streams.
    FIELDS = ("pos_raw", "pos_norm", "spd", "cur", "temp", "err", "cmd_raw", "cmd_norm")

    def __init__(self, capacity=4096):
        self.cap = capacity
        self.t = np.zeros(capacity)
        self.data = {f: np.zeros((capacity, paths.N_MOTORS), np.float32) for f in self.FIELDS}
        self.seq = 0                      # total samples ever written
        self._lock = threading.Lock()

    def push(self, t, sample):
        """sample: {field: array[N_MOTORS]} (missing motors = NaN is fine)."""
        with self._lock:
            i = self.seq % self.cap
            self.t[i] = t
            for f in self.FIELDS:
                self.data[f][i] = sample[f]
            self.seq += 1

    def read_since(self, since_seq, max_samples=512):
        """Samples with seq >= since_seq (clamped to what's still in the ring), oldest first."""
        with self._lock:
            end = self.seq
            start = max(since_seq, end - self.cap, 0)
            start = max(start, end - max_samples)
            n = end - start
            if n <= 0:
                return end, np.zeros(0), {f: np.zeros((0, paths.N_MOTORS)) for f in self.FIELDS}
            idx = np.arange(start, end) % self.cap
            return end, self.t[idx].copy(), {f: self.data[f][idx].copy() for f in self.FIELDS}
