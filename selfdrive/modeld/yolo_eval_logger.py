"""Low-overhead YOLO evaluation statistics.

The realtime thread only updates numeric counters and bounded in-memory samples.
A background thread periodically swaps the active window and writes one JSON
summary, so disk latency cannot block modeld or radard.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WINDOW_SECONDS = 120.0
MAX_ERROR_SAMPLES = 4096
MAX_EVENTS = 100
MAX_TIME_SAMPLES = 32
TIME_SAMPLE_SECONDS = 5.0
YOLO_EVAL_ENABLED = os.getenv("DP_YOLO_EVAL_ENABLED", "0") not in ("0", "false", "False")

DISTANCE_BINS = (
  ("0_5m", 0.0, 5.0),
  ("5_20m", 5.0, 20.0),
  ("20_50m", 20.0, 50.0),
  ("50_100m", 50.0, 100.0),
  ("100m_plus", 100.0, float("inf")),
)


def _finite_float(value: Any, default: float | None = None) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return default
  return result if result == result and abs(result) != float("inf") else default


def _array_value(output: dict[str, Any], name: str, indexes: tuple[int, ...],
                 default: float | None = None) -> float | None:
  try:
    value: Any = output[name]
    for index in indexes:
      value = value[index]
    return _finite_float(value, default)
  except (KeyError, IndexError, TypeError):
    return default


def _percentile(values: deque[float], percentile: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
  return round(float(ordered[index]), 4)


def _summary(values: deque[float]) -> dict[str, Any]:
  if not values:
    return {"count": 0}
  return {
    "count": len(values),
    "mean": round(sum(values) / len(values), 4),
    "median": _percentile(values, 0.5),
    "p90": _percentile(values, 0.9),
    "p95": _percentile(values, 0.95),
    "max": round(max(values), 4),
  }


class _Window:
  def __init__(self, started_wall: float, started_mono: float):
    self.started_wall = started_wall
    self.started_mono = started_mono
    self.frame_count = 0
    self.model_present = 0
    self.yolo_present = 0
    self.yolo_used = 0
    self.model_yolo_pairs = 0
    self.yolo_radar_present = 0
    self.yolo_radar_matched = 0
    self.yolo_radar_rejected = 0
    self.x_errors = deque(maxlen=MAX_ERROR_SAMPLES)
    self.y_errors = deque(maxlen=MAX_ERROR_SAMPLES)
    self.x_relative_errors = deque(maxlen=MAX_ERROR_SAMPLES)
    self.radar_x_errors = deque(maxlen=MAX_ERROR_SAMPLES)
    self.radar_y_errors = deque(maxlen=MAX_ERROR_SAMPLES)
    self.distance_bins = {
      name: {
        "count": 0,
        "x_errors": deque(maxlen=MAX_ERROR_SAMPLES // 4),
        "y_errors": deque(maxlen=MAX_ERROR_SAMPLES // 4),
      }
      for name, _, _ in DISTANCE_BINS
    }
    self.events: list[dict[str, Any]] = []
    self.time_samples: list[dict[str, Any]] = []
    self.last_sample_mono = 0.0

  def add_event(self, event: dict[str, Any]) -> None:
    if len(self.events) < MAX_EVENTS:
      self.events.append(event)

  def add_time_sample(self, sample: dict[str, Any], now_mono: float) -> None:
    if now_mono - self.last_sample_mono >= TIME_SAMPLE_SECONDS or not self.time_samples:
      if len(self.time_samples) < MAX_TIME_SAMPLES:
        self.time_samples.append(sample)
      self.last_sample_mono = now_mono

  def distance_bin(self, distance: float) -> str:
    for name, lower, upper in DISTANCE_BINS:
      if lower <= distance < upper:
        return name
    return DISTANCE_BINS[-1][0]

  def to_json(self, ended_wall: float, ended_mono: float, process: str) -> dict[str, Any]:
    bins: dict[str, Any] = {}
    for name, values in self.distance_bins.items():
      bins[name] = {
        "count": values["count"],
        "x_error_m": _summary(values["x_errors"]),
        "y_error_m": _summary(values["y_errors"]),
      }

    return {
      "schema": 1,
      "process": process,
      "start_time_utc": datetime.fromtimestamp(self.started_wall, timezone.utc).isoformat(),
      "end_time_utc": datetime.fromtimestamp(ended_wall, timezone.utc).isoformat(),
      "duration_s": round(max(0.0, ended_mono - self.started_mono), 3),
      "frames": {
        "total": self.frame_count,
        "model_lead_present": self.model_present,
        "yolo_present": self.yolo_present,
        "yolo_used": self.yolo_used,
        "model_yolo_pairs": self.model_yolo_pairs,
      },
      "radar": {
        "yolo_present": self.yolo_radar_present,
        "matched": self.yolo_radar_matched,
        "rejected": self.yolo_radar_rejected,
      },
      "model_yolo_error_m": {
        "x": _summary(self.x_errors),
        "y": _summary(self.y_errors),
        "x_relative": _summary(self.x_relative_errors),
      },
      "yolo_radar_error_m": {
        "x": _summary(self.radar_x_errors),
        "y": _summary(self.radar_y_errors),
      },
      "distance_bins": bins,
      "time_samples": self.time_samples,
      "events": self.events,
    }


class YoloEvalLogger:
  """Aggregate YOLO evaluation data without blocking the caller."""

  def __init__(self, process: str, output_dir: str | Path | None = None,
               window_seconds: float = WINDOW_SECONDS, enabled: bool | None = None):
    self.process = process
    self.enabled = YOLO_EVAL_ENABLED if enabled is None else bool(enabled)
    self.window_seconds = max(1.0, float(window_seconds))
    configured_dir = output_dir or os.getenv("DP_YOLO_EVAL_LOG_DIR")
    self.output_dir = Path(configured_dir) if configured_dir else Path.cwd() / "yolo_eval_logs"
    self._lock = threading.Lock()
    self._closed = False
    if not self.enabled:
      self._window = None
      self._stop = threading.Event()
      self._thread = None
      return

    now_wall = time.time()
    now_mono = time.monotonic()
    self._window = _Window(now_wall, now_mono)
    self._stop = threading.Event()
    self._thread = threading.Thread(
      target=self._writer_loop,
      name=f"{process}_yolo_eval_writer",
      daemon=True,
    )
    self._thread.start()
    atexit.register(self.close)

  def _new_window_locked(self) -> _Window:
    assert self._window is not None
    now_wall = time.time()
    now_mono = time.monotonic()
    old_window = self._window
    self._window = _Window(now_wall, now_mono)
    return old_window

  def _writer_loop(self) -> None:
    while not self._stop.wait(self.window_seconds):
      self.flush()

  def _write_window(self, window: _Window) -> None:
    try:
      self.output_dir.mkdir(parents=True, exist_ok=True)
      ended_wall = time.time()
      ended_mono = time.monotonic()
      payload = window.to_json(ended_wall, ended_mono, self.process)
      stamp = datetime.fromtimestamp(window.started_wall, timezone.utc).strftime("%Y%m%d_%H%M%S")
      filename = self.output_dir / f"yolo_eval_{self.process}_{stamp}_{os.getpid()}.json"
      temporary = filename.with_suffix(".json.tmp")
      with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=True, separators=(",", ":"))
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
      os.replace(temporary, filename)
    except Exception:
      # Evaluation must never affect driving. A failed write is intentionally
      # ignored; the next window will try again.
      pass

  def flush(self) -> None:
    if not self.enabled:
      return
    with self._lock:
      window = self._new_window_locked()
    self._write_window(window)

  def close(self) -> None:
    if self._closed or not self.enabled:
      return
    self._closed = True
    self._stop.set()
    if threading.current_thread() is not self._thread:
      self._thread.join(timeout=0.2)
    self.flush()

  def record_model_frame(self, frame_id: int, model_output: dict[str, Any],
                         yolo_lead: Any | None, yolo_used: bool,
                         rejection_reason: str | None = None,
                         model_lead: dict[str, float | None] | None = None) -> None:
    if not self.enabled:
      return
    now_mono = time.monotonic()
    model_prob = (
      _finite_float(model_lead.get("prob"), 0.0)
      if model_lead is not None else _array_value(model_output, "lead_prob", (0, 0), 0.0)
    ) or 0.0
    model_x = (
      _finite_float(model_lead.get("x"))
      if model_lead is not None else _array_value(model_output, "lead", (0, 0, 0))
    )
    model_y = (
      _finite_float(model_lead.get("y"))
      if model_lead is not None else _array_value(model_output, "lead", (0, 0, 1))
    )
    yolo_prob = _finite_float(getattr(yolo_lead, "prob", None))
    yolo_x = _finite_float(getattr(yolo_lead, "x", None))
    yolo_y = _finite_float(getattr(yolo_lead, "y", None))
    model_has_lead = model_prob > 0.5
    has_yolo = yolo_lead is not None and yolo_x is not None and yolo_y is not None

    with self._lock:
      window = self._window
      window.frame_count += 1
      window.model_present += int(model_has_lead)
      window.yolo_present += int(has_yolo)
      window.yolo_used += int(yolo_used)
      if has_yolo and model_x is not None and model_y is not None:
        window.model_yolo_pairs += 1
        x_error = abs(yolo_x - model_x)
        y_error = abs(yolo_y - model_y)
        window.x_errors.append(x_error)
        window.y_errors.append(y_error)
        window.x_relative_errors.append(x_error / max(abs(model_x), 1.0))
        bin_values = window.distance_bins[window.distance_bin(yolo_x)]
        bin_values["count"] += 1
        bin_values["x_errors"].append(x_error)
        bin_values["y_errors"].append(y_error)
        if x_error > max(5.0, abs(model_x) * 0.35) or y_error > 1.25:
          window.add_event({
            "type": "model_yolo_large_error",
            "frame_id": int(frame_id),
            "yolo_x_m": round(yolo_x, 3),
            "yolo_y_m": round(yolo_y, 3),
            "model_x_m": round(model_x, 3),
            "model_y_m": round(model_y, 3),
            "x_error_m": round(x_error, 3),
            "y_error_m": round(y_error, 3),
          })
      if has_yolo and not yolo_used and rejection_reason:
        window.add_event({
          "type": "yolo_not_used",
          "frame_id": int(frame_id),
          "reason": str(rejection_reason),
          "yolo_x_m": round(yolo_x, 3),
          "yolo_y_m": round(yolo_y, 3),
          "yolo_prob": round(yolo_prob or 0.0, 4),
        })
      window.add_time_sample({
        "frame_id": int(frame_id),
        "yolo_present": bool(has_yolo),
        "yolo_used": bool(yolo_used),
        "model_prob": round(model_prob, 4),
        "model_x_m": None if model_x is None else round(model_x, 3),
        "model_y_m": None if model_y is None else round(model_y, 3),
        "yolo_x_m": None if yolo_x is None else round(yolo_x, 3),
        "yolo_y_m": None if yolo_y is None else round(yolo_y, 3),
      }, now_mono)

  def record_radar_frame(self, frame_id: int, yolo_present: bool,
                         yolo_lead: Any, radar_lead: Any) -> None:
    if not self.enabled:
      return
    if not yolo_present:
      return
    matched = bool(getattr(radar_lead, "status", False) and getattr(radar_lead, "radar", False))
    yolo_x = _finite_float(yolo_lead.x[0] if len(yolo_lead.x) else None)
    yolo_y = _finite_float(yolo_lead.y[0] if len(yolo_lead.y) else None)
    radar_x = _finite_float(getattr(radar_lead, "dRel", None))
    radar_y = _finite_float(getattr(radar_lead, "yRel", None))
    with self._lock:
      window = self._window
      window.yolo_radar_present += 1
      window.yolo_radar_matched += int(matched)
      window.yolo_radar_rejected += int(not matched)
      if matched and yolo_x is not None and yolo_y is not None and radar_x is not None and radar_y is not None:
        radar_x_error = abs(yolo_x - (radar_x + 1.52))
        radar_y_error = abs(-yolo_y - radar_y)
        window.radar_x_errors.append(radar_x_error)
        window.radar_y_errors.append(radar_y_error)
        if radar_x_error > max(5.0, abs(yolo_x) * 0.35) or radar_y_error > 1.25:
          window.add_event({
            "type": "yolo_radar_large_error",
            "frame_id": int(frame_id),
            "yolo_x_m": round(yolo_x, 3),
            "yolo_y_m": round(yolo_y, 3),
            "radar_x_camera_m": round(radar_x + 1.52, 3),
            "radar_y_m": round(radar_y, 3),
            "x_error_m": round(radar_x_error, 3),
            "y_error_m": round(radar_y_error, 3),
          })
      if not matched:
        window.add_event({
          "type": "yolo_radar_rejected",
          "frame_id": int(frame_id),
          "yolo_x_m": yolo_x,
          "yolo_y_m": yolo_y,
          "radar_x_m": radar_x,
          "radar_y_m": radar_y,
        })
