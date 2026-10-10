import json
import time
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.modeld.yolo_eval_logger import YoloEvalLogger


def make_model_output(prob: float = 0.8, x: float = 20.0, y: float = 0.0):
  lead = np.zeros((1, 3, 6, 4), dtype=np.float32)
  lead[0, 0, 0] = [x, y, 0.0, 0.0]
  return {
    "lead_prob": np.array([[prob]], dtype=np.float32),
    "lead": lead,
  }


def test_yolo_eval_logger_is_disabled_without_environment_flag(tmp_path):
  logger = YoloEvalLogger("modeld", tmp_path, enabled=False)
  logger.record_model_frame(1, make_model_output(), SimpleNamespace(prob=0.9, x=21.0, y=0.1), True)
  logger.flush()
  assert list(tmp_path.iterdir()) == []


def test_yolo_eval_logger_writes_aggregated_json(tmp_path):
  logger = YoloEvalLogger("modeld", tmp_path, window_seconds=120.0, enabled=True)
  logger.record_model_frame(1, make_model_output(), SimpleNamespace(prob=0.9, x=21.0, y=0.1), True)
  logger.flush()
  logger.close()

  files = list(tmp_path.glob("yolo_eval_modeld_*.json"))
  assert len(files) == 1
  data = json.loads(files[0].read_text(encoding="utf-8"))
  assert data["frames"]["total"] == 1
  assert data["frames"]["yolo_used"] == 1
  assert data["model_yolo_error_m"]["x"]["count"] == 1


def test_yolo_eval_logger_tracks_radar_match_and_coordinate_sign(tmp_path):
  logger = YoloEvalLogger("radard", tmp_path, enabled=True)
  yolo_lead = SimpleNamespace(x=[20.0], y=[0.5])
  radar_lead = SimpleNamespace(status=True, radar=True, dRel=18.48, yRel=-0.5)
  logger.record_radar_frame(1, True, yolo_lead, radar_lead)
  logger.flush()
  logger.close()

  files = list(tmp_path.glob("yolo_eval_radard_*.json"))
  assert len(files) == 1
  data = json.loads(files[0].read_text(encoding="utf-8"))
  assert data["radar"]["matched"] == 1
  assert data["yolo_radar_error_m"]["x"]["p95"] == 0.0
  assert data["yolo_radar_error_m"]["y"]["p95"] == 0.0
