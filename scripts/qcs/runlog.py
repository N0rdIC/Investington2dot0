"""Logging for the backtest run.

Produces two artifacts you send back to me:

  1. run_log_<timestamp>.txt   -- everything printed, plus environment + errors
  2. run_results_<timestamp>.json -- the structured numbers, so I analyse the
     actual values instead of parsing prose out of a console dump.

Usage in a script:

    from qcs.runlog import RunLogger
    log = RunLogger("run")          # start capturing
    ...                             # normal prints are teed to the file
    log.record("null_test", {...})  # add structured results
    log.close()                     # writes both files, prints their paths
"""
from __future__ import annotations

import json
import platform
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


class _Tee:
    """Duplicate a stream to the console and a file."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def _jsonable(obj):
    """Make numpy / pandas values JSON-serialisable and NaN-safe."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    # last resort: stringify anything exotic (Timestamp, etc.) rather than crash
    return str(obj)


class RunLogger:
    def __init__(self, tag: str = "run", out_dir: str = "logs"):
        self.ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.txt_path = self.dir / f"{tag}_log_{self.ts}.txt"
        self.json_path = self.dir / f"{tag}_results_{self.ts}.json"

        self._fh = open(self.txt_path, "w")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._stdout, self._fh)
        sys.stderr = _Tee(self._stderr, self._fh)

        self.results: dict = {"_meta": self._env()}
        self._banner()

    def _env(self) -> dict:
        env = {
            "timestamp_utc": self.ts,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        for mod in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "yfinance"):
            try:
                m = __import__(mod)
                env[mod] = getattr(m, "__version__", "?")
            except Exception:
                env[mod] = "NOT INSTALLED"
        return env

    def _banner(self):
        print("=" * 66)
        print(f"RUN LOG  {self.ts} UTC")
        print("=" * 66)
        for k, v in self.results["_meta"].items():
            print(f"  {k:<14} {v}")
        print("=" * 66 + "\n")

    def section(self, name: str):
        print("\n" + "#" * 66)
        print(f"#  {name}")
        print("#" * 66)

    def record(self, key: str, value):
        """Store a structured result under `key` for the JSON dump."""
        self.results[key] = _jsonable(value)

    def error(self, exc: BaseException):
        print("\n" + "!" * 66)
        print("ERROR — run did not complete")
        print("!" * 66)
        traceback.print_exc()
        self.results["_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    def close(self):
        with open(self.json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._fh.close()
        print(f"\nLog written:\n  {self.txt_path}\n  {self.json_path}")
        print("\nSend me BOTH files (or just paste the .txt).")
