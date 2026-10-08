import threading
import time

import numpy as np

from openpilot.selfdrive.modeld import yolo


def make_model_output(model_lead_prob: float = 0.0) -> dict[str, np.ndarray]:
  return {
    "lead": np.zeros((1, yolo.ModelConstants.LEAD_TRAJ_LEN, yolo.ModelConstants.LEAD_WIDTH), dtype=np.float32),
    "lead_stds": np.zeros((1, yolo.ModelConstants.LEAD_TRAJ_LEN, yolo.ModelConstants.LEAD_WIDTH), dtype=np.float32),
    "lead_prob": np.array([[model_lead_prob]], dtype=np.float32),
    "plan": np.zeros((1, yolo.ModelConstants.IDX_N, yolo.ModelConstants.PLAN_WIDTH), dtype=np.float32),
    "lane_lines": np.zeros((1, 4, yolo.ModelConstants.IDX_N, yolo.ModelConstants.LANE_LINES_WIDTH), dtype=np.float32),
    "lane_lines_prob": np.array([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32),
  }


def make_lane_model_output(model_lead_prob: float = 0.0) -> dict[str, np.ndarray]:
  output = make_model_output(model_lead_prob)
  output["lane_lines"][0, 1, :, 0] = -2.0
  output["lane_lines"][0, 2, :, 0] = 2.0
  return output


def test_yolo_cpu_preprocess_keeps_expected_tensor_layout():
  class FakeVisionBuf:
    width = 200
    height = 100
    stride = 200
    uv_offset = width * height
    data = np.full(width * height * 3 // 2, 128, dtype=np.uint8)

  tensor, scale, dx, dy = yolo.yolo_preprocess_nv12_cpu(FakeVisionBuf())

  assert tensor.shape == (1, 3, yolo.YOLO_INPUT_H, yolo.YOLO_INPUT_W)
  assert tensor.dtype == np.float32
  assert scale == 2.56
  assert dx == 0
  assert dy == 0


def test_yolo_cpu_preprocess_handles_camera_nv12_dimensions():
  class FakeVisionBuf:
    width = 1920
    height = 1080
    stride = 1920
    uv_offset = width * height
    data = np.full(width * height * 3 // 2, 128, dtype=np.uint8)

  tensor, _, _, _ = yolo.yolo_preprocess_nv12_cpu(FakeVisionBuf())

  assert tensor.shape == (1, 3, yolo.YOLO_INPUT_H, yolo.YOLO_INPUT_W)
  assert tensor.dtype == np.float32


def test_yolo_preprocess_buffers_reuse_double_input_buffers():
  class FakeVisionBuf:
    width = 200
    height = 100
    stride = 200
    uv_offset = width * height
    data = np.full(width * height * 3 // 2, 128, dtype=np.uint8)

  buffers = yolo.YoloPreprocessBuffers(FakeVisionBuf.width, FakeVisionBuf.height)
  first, _, _, _ = buffers.process_cpu(FakeVisionBuf())
  second, _, _, _ = buffers.process_cpu(FakeVisionBuf())
  third, _, _, _ = buffers.process_cpu(FakeVisionBuf())

  assert first is buffers.input_buffers[0]
  assert second is buffers.input_buffers[1]
  assert third is buffers.input_buffers[0]
  assert first.shape == (1, 3, yolo.YOLO_INPUT_H, yolo.YOLO_INPUT_W)
  assert np.allclose(first, second)


def test_yolo_opencl_preprocess_uses_only_small_transformed_frame():
  class FakeVisionBuf:
    width = 1928
    height = 1208

  class FakeYoloFrame:
    def __init__(self):
      self.buf = None
      self.projection = None

    def process(self, buf, projection, left, top, right, bottom):
      self.buf = buf
      self.projection = projection
      self.letterbox = (left, top, right, bottom)
      # The native wrapper returns only the final normalized NCHW tensor.
      return np.full(yolo.YOLO_INPUT_W * yolo.YOLO_INPUT_H * 3, 114.0 / 255.0, dtype=np.float32)

  runner = yolo.RoadYoloRunner(enabled=True)
  runner.yolo_frame = FakeYoloFrame()
  buf = FakeVisionBuf()
  tensor, _, dx, dy = runner._preprocess_opencl(buf)

  assert runner.yolo_frame.buf is buf
  assert runner.yolo_frame.projection.shape == (9,)
  assert runner.yolo_frame.letterbox == (dx, dy, yolo.YOLO_INPUT_W - dx, yolo.YOLO_INPUT_H - dy)
  assert tensor.shape == (1, 3, yolo.YOLO_INPUT_H, yolo.YOLO_INPUT_W)
  assert tensor.dtype == np.float32
  assert tensor.flags.c_contiguous
  assert dx > 0
  assert dy == 0
  assert np.allclose(tensor, 114.0 / 255.0)


def test_yolo_nms_suppresses_overlapping_box():
  boxes = np.array([[0, 0, 100, 100], [10, 10, 100, 100], [300, 300, 20, 20]], dtype=np.float32)
  scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)

  keep = yolo.yolo_nms(boxes, scores, 0.45)

  assert keep == [1, 2]


def test_yolo_cpu_affinity_uses_requested_core(monkeypatch):
  monkeypatch.setattr(yolo.os, "name", "posix", raising=False)
  monkeypatch.setattr(yolo.os, "sched_getaffinity", lambda _: {4, *yolo.YOLO_CPU_CORES})
  monkeypatch.setattr(yolo.threading, "get_native_id", lambda: 1234)
  selected_cores = []
  selected_tids = []
  monkeypatch.setattr(yolo.os, "sched_setaffinity", lambda tid, cores: (selected_tids.append(tid), selected_cores.append(cores)))

  assert yolo.set_yolo_cpu_affinity() == yolo.YOLO_CPU_CORES
  assert selected_tids == [1234]
  assert selected_cores == [set(yolo.YOLO_CPU_CORES)]


def test_yolo26_postprocess_decodes_class_scores_without_objectness():
  runner = yolo.RoadYoloRunner(enabled=True)
  output = np.zeros((1, 84, 1), dtype=np.float32)
  output[0, 0:4, 0] = [256.0, 128.0, 100.0, 60.0]
  output[0, 4 + 2, 0] = 0.9

  detections = runner._postprocess([output], 1.0, 0, 0, 512, 256)

  assert detections == [(2, 0.9, (206, 98, 100, 60))]


def test_yolo26_postprocess_keeps_only_lane_obstacle_classes():
  runner = yolo.RoadYoloRunner(enabled=True)
  output = np.zeros((1, 84, 2), dtype=np.float32)
  output[0, 0:4, 0] = [100.0, 100.0, 40.0, 40.0]
  output[0, 4 + 4, 0] = 0.95
  output[0, 0:4, 1] = [300.0, 120.0, 40.0, 40.0]
  output[0, 4 + 16, 1] = 0.9

  detections = runner._postprocess([output], 1.0, 0, 0, 512, 256)

  assert detections == [(16, 0.9, (280, 100, 40, 40))]


def test_detection_uses_camera_intrinsics_for_lead_position():
  intrinsics = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float32)

  lead = yolo.yolo_detection_to_lead((2, 0.8, (450, 100, 100, 100)), intrinsics, (1000, 800), 1000)

  assert lead is not None
  assert lead.prob == 0.8
  assert lead.x == 18.0
  assert lead.y == 0.0


def test_yolo_lead_must_be_inside_model_lane():
  model_output = make_lane_model_output()

  assert yolo.yolo_lead_in_model_lane(model_output, yolo.YoloLead(0.8, 20.0, 0.0, 1, 0, time.monotonic()))
  assert not yolo.yolo_lead_in_model_lane(model_output, yolo.YoloLead(0.8, 20.0, 2.5, 1, 0, time.monotonic()))


def test_yolo_lead_is_rejected_when_lane_lines_are_unavailable():
  model_output = make_model_output()
  model_output["lane_lines_prob"][0] = 0.0
  lead = yolo.YoloLead(0.8, 20.0, 5.0, 1, 0, time.monotonic())

  assert not yolo.yolo_lead_in_model_lane(model_output, lead)


def test_model_lead_has_priority_by_default(monkeypatch):
  monkeypatch.setattr(yolo, "YOLO_USE_ONLY_LEAD", False)
  model_output = make_lane_model_output(0.8)
  lead = yolo.YoloLead(0.9, 20.0, 0.0, 1, 0, time.monotonic())

  assert not yolo.apply_yolo_lead_if_needed(model_output, lead)
  assert model_output["lead_prob"][0, 0] == 0.8


def test_yolo_can_replace_model_lead_in_test_mode(monkeypatch):
  monkeypatch.setattr(yolo, "YOLO_USE_ONLY_LEAD", True)
  model_output = make_lane_model_output(0.8)
  lead = yolo.YoloLead(0.9, 20.0, 0.0, 1, 0, time.monotonic())

  assert yolo.apply_yolo_lead_if_needed(model_output, lead)
  assert model_output["lead_prob"][0, 0] == 0.9
  assert np.all(model_output["lead"][0, :, 0] == 20.0)
  assert np.all(model_output["lead"][0, :, 1] == 0.0)
  assert np.all(model_output["lead_stds"][0, :, 0] == 5.0)


def test_runner_waits_for_requested_frame():
  runner = yolo.RoadYoloRunner(enabled=True)
  result: list[yolo.YoloLead | None] = []
  lead = yolo.YoloLead(0.8, 20.0, 0.0, 7, 0, time.monotonic())

  waiter = threading.Thread(target=lambda: result.append(runner.wait_for_frame(7)))
  waiter.start()
  time.sleep(0.01)
  assert waiter.is_alive()

  runner._set_latest_lead(lead, 7)
  waiter.join(timeout=1.0)

  assert not waiter.is_alive()
  assert result == [lead]


def test_runner_timeout_does_not_block_modeld():
  runner = yolo.RoadYoloRunner(enabled=True)
  started = time.monotonic()

  assert runner.wait_for_frame(99, timeout_ms=5.0) is None
  assert (time.monotonic() - started) < 0.2


def test_runner_timeout_drops_pending_task():
  class FakeVisionBuf:
    width = yolo.YOLO_INPUT_W
    height = yolo.YOLO_INPUT_H

  runner = yolo.RoadYoloRunner(enabled=True)
  runner.pending_task = yolo.YoloTask(
    frame_id=99,
    timestamp_eof=0,
    buf=FakeVisionBuf(),
    submitted_t=time.perf_counter(),
  )

  assert runner.wait_for_frame(99, timeout_ms=0.0) is None
  assert runner.pending_task is None


def test_runner_timeout_waits_for_active_preprocess_to_finish():
  runner = yolo.RoadYoloRunner(enabled=True)
  result: list[yolo.YoloLead | None] = []

  with runner.condition:
    runner.active_preprocess_frame_id = 99

  waiter = threading.Thread(target=lambda: result.append(runner.wait_for_frame(99, timeout_ms=0.0)))
  waiter.start()
  time.sleep(0.01)
  assert waiter.is_alive()

  with runner.condition:
    runner.active_preprocess_frame_id = -1
    runner.condition.notify_all()
  waiter.join(timeout=1.0)

  assert not waiter.is_alive()
  assert result == [None]


def test_runner_ignores_late_result_after_timeout():
  runner = yolo.RoadYoloRunner(enabled=True)
  lead = yolo.YoloLead(0.8, 20.0, 0.0, 99, 0, time.monotonic())

  assert runner.wait_for_frame(99, timeout_ms=0.0) is None
  runner._set_latest_lead(lead, 99)

  assert runner.get_latest_lead() is None
  assert runner.wait_for_frame(99, timeout_ms=0.0) is None
