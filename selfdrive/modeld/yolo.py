import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from msgq.visionipc import VisionBuf

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, YoloFrame


YOLO_ENABLED = os.getenv("DP_YOLO_ENABLED", "1") not in ("0", "false", "False")
# Test switch: when enabled, a valid YOLO lead replaces modelV2 lead[0].
YOLO_USE_ONLY_LEAD = os.getenv("DP_YOLO_USE_ONLY_LEAD", "0") not in ("0", "false", "False")
YOLO_MODEL_PATH = Path(__file__).parent / "models/yolo26n.onnx"

YOLO_INPUT_H = 256
YOLO_INPUT_W = 512
YOLO_CONF_THRESHOLD = 0.25
YOLO_NMS_THRESHOLD = 0.45
YOLO_LEAD_MAX_AGE = 0.25
YOLO_WAIT_MS = float(os.getenv("DP_YOLO_WAIT_MS", "35.0"))
YOLO_SLOW_FRAME_MS = 49.0
YOLO_INTRA_OP_THREADS = 2
# The driving model owns the shared OpenCL device. Letting YOLO enqueue work
# on it can delay model.run enough to drop camera frames, so keep YOLO CPU
# preprocessing by default. DP_YOLO_OPENCL=1 remains available for profiling.
YOLO_USE_OPENCL = os.getenv("DP_YOLO_OPENCL", "1") not in ("0", "false", "False")
YOLO_CPU_CORES = tuple(int(core.strip()) for core in os.getenv("DP_YOLO_CPU_CORES", "5,9").split(",") if core.strip())
YOLO_DEFAULT_REAL_WIDTH = 0.5
YOLO_LANE_PROB_THRESHOLD = 0.5
YOLO_LANE_MARGIN = 0.1
YOLO_LANE_OBSTACLE_CLASS_IDS = {0, 1, 2, 3, 5, 7, 16, 17, 18, 19}
YOLO_REAL_WIDTH_BY_CLASS = [
  0.50, 0.60, 1.80, 0.80, 2.50, 2.50, 3.00, 2.50,
  1.00, 0.50, 0.50, 0.60, 0.20, 0.60, 0.40, 0.40,
  0.50, 0.40, 0.40, 0.40, 1.00, 0.60, 0.30, 0.50,
  1.00, 0.50, 1.00, 0.30, 1.00, 0.50, 0.30, 0.50,
  0.30, 0.30, 0.50, 0.50, 0.50, 0.50, 0.50, 0.30,
  0.30, 0.20, 0.20, 0.20, 0.20, 0.30, 0.30, 0.30,
  0.30, 0.30, 0.30, 0.30, 0.10, 0.30, 0.20, 0.30,
  0.50, 0.80, 0.50, 0.60, 1.00, 1.00, 0.60, 0.50,
  0.30, 0.40, 0.20, 0.20, 0.50, 0.60, 0.60, 0.50,
  0.50, 0.30, 0.40, 0.30, 0.20, 0.30, 0.30, 0.20,
]

_last_yolo_rejection_reason: str | None = None
_last_yolo_rejection_log_t = 0.0


def get_last_yolo_rejection_reason() -> str | None:
  return _last_yolo_rejection_reason


@dataclass
class YoloLead:
  prob: float
  x: float
  y: float
  frame_id: int
  timestamp_eof: int
  created_t: float


@dataclass
class YoloTask:
  frame_id: int
  timestamp_eof: int
  buf: VisionBuf
  submitted_t: float


def yolo_letterbox_geometry(src_w: int, src_h: int) -> tuple[float, int, int, int, int]:
  scale = min(YOLO_INPUT_W / src_w, YOLO_INPUT_H / src_h)
  resized_w = int(src_w * scale) & ~1
  resized_h = int(src_h * scale) & ~1
  resized_w = max(2, resized_w)
  resized_h = max(2, resized_h)
  dx = (YOLO_INPUT_W - resized_w) // 2
  dy = (YOLO_INPUT_H - resized_h) // 2
  return scale, resized_w, resized_h, dx, dy


def yolo_nv12_to_tensor(nv12: np.ndarray, width: int, height: int,
                        scale: float, dx: int, dy: int) -> tuple[np.ndarray, float, int, int]:
  """Convert a small, already-resized NV12 image to a letterboxed RGB tensor."""
  y = nv12[:width * height].reshape(height, width).astype(np.float32)
  uv = nv12[width * height:].reshape(height // 2, width // 2, 2).astype(np.float32)
  # Upsample chroma with indexed views instead of allocating two repeat chains.
  u = uv[:, :, 0][::, :][np.arange(height) // 2][:, np.arange(width) // 2]
  v = uv[:, :, 1][::, :][np.arange(height) // 2][:, np.arange(width) // 2]

  # BT.601 limited-range NV12 conversion. Camera pixels stay on GPU until now.
  y = np.maximum(y - 16.0, 0.0) * 1.164383
  u -= 128.0
  v -= 128.0
  rgb = np.empty((height, width, 3), dtype=np.float32)
  rgb[..., 0] = y + 1.596027 * v
  rgb[..., 1] = y - 0.391762 * u - 0.812968 * v
  rgb[..., 2] = y + 2.017232 * u
  np.clip(rgb, 0.0, 255.0, out=rgb)

  if width == YOLO_INPUT_W and height == YOLO_INPUT_H:
    # transform.cl clamps samples outside the source image. Replace those
    # pixels with the YOLO letterbox color after the small GPU download.
    if dy:
      rgb[:dy] = 114.0
      rgb[YOLO_INPUT_H - dy:] = 114.0
    if dx:
      rgb[:, :dx] = 114.0
      rgb[:, YOLO_INPUT_W - dx:] = 114.0
    return np.transpose(rgb, (2, 0, 1))[np.newaxis] / 255.0, scale, dx, dy

  letterboxed = np.full((YOLO_INPUT_H, YOLO_INPUT_W, 3), 114.0, dtype=np.float32)
  letterboxed[dy:dy + height, dx:dx + width] = rgb
  return np.transpose(letterboxed, (2, 0, 1))[np.newaxis] / 255.0, scale, dx, dy


def yolo_opencl_projection(src_w: int, src_h: int) -> tuple[np.ndarray, float, int, int]:
  scale, resized_w, resized_h, dx, dy = yolo_letterbox_geometry(src_w, src_h)
  projection = np.array([
    [src_w / resized_w, 0.0, -dx * src_w / resized_w],
    [0.0, src_h / resized_h, -dy * src_h / resized_h],
    [0.0, 0.0, 1.0],
  ], dtype=np.float32)
  return projection, scale, dx, dy


class YoloPreprocessBuffers:
  """Reusable CPU/OpenCL staging buffers for one camera resolution."""

  def __init__(self, src_w: int, src_h: int):
    self.src_w = src_w
    self.src_h = src_h
    self.scale, self.resized_w, self.resized_h, self.dx, self.dy = yolo_letterbox_geometry(src_w, src_h)
    self.projection, _, _, _ = yolo_opencl_projection(src_w, src_h)
    self.input_buffers = [
      np.empty((1, 3, YOLO_INPUT_H, YOLO_INPUT_W), dtype=np.float32),
      np.empty((1, 3, YOLO_INPUT_H, YOLO_INPUT_W), dtype=np.float32),
    ]
    self.input_index = 0

    self.x_indices = np.linspace(0, src_w - 1, self.resized_w).astype(np.intp)
    self.y_indices = np.linspace(0, src_h - 1, self.resized_h).astype(np.intp)
    self.uv_x_indices = np.linspace(0, src_w // 2 - 1, self.resized_w // 2).astype(np.intp) * 2
    self.uv_y_indices = self.y_indices[::2] // 2
    self.uv_full_y_indices = np.arange(self.resized_h) // 2
    self.uv_full_x_indices = np.arange(self.resized_w) // 2
    self.nv12 = np.empty(self.resized_w * self.resized_h * 3 // 2, dtype=np.uint8)
    self.y_small = self.nv12[:self.resized_w * self.resized_h].reshape(self.resized_h, self.resized_w)
    self.uv_small = self.nv12[self.resized_w * self.resized_h:].reshape(self.resized_h // 2, self.resized_w)
    self.y_float = np.empty((self.resized_h, self.resized_w), dtype=np.float32)
    self.u_half = np.empty((self.resized_h // 2, self.resized_w // 2), dtype=np.float32)
    self.v_half = np.empty_like(self.u_half)
    self.u_full = np.empty((self.resized_h, self.resized_w), dtype=np.float32)
    self.v_full = np.empty_like(self.u_full)
    self.rgb = np.empty((self.resized_h, self.resized_w, 3), dtype=np.float32)
    self.letterboxed = np.empty((YOLO_INPUT_H, YOLO_INPUT_W, 3), dtype=np.float32)
    self.nchw_scale = 1.0 / 255.0

  def next_input(self) -> np.ndarray:
    output = self.input_buffers[self.input_index]
    self.input_index = (self.input_index + 1) % len(self.input_buffers)
    return output

  def _finish_tensor(self, output: np.ndarray) -> np.ndarray:
    # Keep the letterbox allocation alive and copy directly into NCHW memory.
    self.letterboxed.fill(114.0)
    self.letterboxed[self.dy:self.dy + self.resized_h, self.dx:self.dx + self.resized_w] = self.rgb
    np.multiply(np.transpose(self.letterboxed, (2, 0, 1)), self.nchw_scale, out=output[0])
    return output

  def process_cpu(self, buf: VisionBuf) -> tuple[np.ndarray, float, int, int]:
    data = np.asarray(buf.data)
    y = data[:buf.uv_offset].reshape((buf.height, buf.stride))[:, :buf.width]
    uv = data[buf.uv_offset:buf.uv_offset + buf.stride * buf.height // 2].reshape(
      (buf.height // 2, buf.stride),
    )[:, :buf.width]

    self.y_small[:] = y[np.ix_(self.y_indices, self.x_indices)]
    uv_rows = uv[self.uv_y_indices]
    self.uv_small[:, 0::2] = uv_rows[:, self.uv_x_indices]
    self.uv_small[:, 1::2] = uv_rows[:, self.uv_x_indices + 1]

    np.copyto(self.y_float, self.y_small, casting="unsafe")
    np.subtract(self.y_float, 16.0, out=self.y_float)
    np.maximum(self.y_float, 0.0, out=self.y_float)
    self.y_float *= 1.164383

    uv_view = self.uv_small.reshape(self.resized_h // 2, self.resized_w // 2, 2)
    np.copyto(self.u_half, uv_view[..., 0], casting="unsafe")
    np.copyto(self.v_half, uv_view[..., 1], casting="unsafe")
    self.u_half -= 128.0
    self.v_half -= 128.0
    self.u_full[:] = self.u_half[self.uv_full_y_indices][:, self.uv_full_x_indices]
    self.v_full[:] = self.v_half[self.uv_full_y_indices][:, self.uv_full_x_indices]

    self.rgb[..., 0] = self.y_float + 1.596027 * self.v_full
    self.rgb[..., 1] = self.y_float - 0.391762 * self.u_full - 0.812968 * self.v_full
    self.rgb[..., 2] = self.y_float + 2.017232 * self.u_full
    np.clip(self.rgb, 0.0, 255.0, out=self.rgb)
    return self._finish_tensor(self.next_input()), self.scale, self.dx, self.dy

  def process_opencl(self, yolo_frame: YoloFrame, buf: VisionBuf) -> tuple[np.ndarray, float, int, int]:
    tensor = yolo_frame.process(
      buf, self.projection.reshape(-1), self.dx, self.dy,
      self.dx + self.resized_w, self.dy + self.resized_h,
    )
    output = self.next_input()
    np.copyto(output, tensor.reshape(output.shape), casting="unsafe")
    return output, self.scale, self.dx, self.dy


def yolo_preprocess_nv12_cpu(buf: VisionBuf) -> tuple[np.ndarray, float, int, int]:
  """CPU fallback. It avoids OpenCV and remains available if OpenCL fails."""
  scale, resized_w, resized_h, dx, dy = yolo_letterbox_geometry(buf.width, buf.height)
  data = np.asarray(buf.data)
  y = data[:buf.uv_offset].reshape((buf.height, buf.stride))[:, :buf.width]
  uv = data[buf.uv_offset:buf.uv_offset + buf.stride * buf.height // 2].reshape((buf.height // 2, buf.stride))[:, :buf.width]
  x_indices = np.linspace(0, buf.width - 1, resized_w).astype(np.intp)
  y_indices = np.linspace(0, buf.height - 1, resized_h).astype(np.intp)
  uv_x_indices = np.linspace(0, buf.width // 2 - 1, resized_w // 2).astype(np.intp) * 2
  y_small = y[y_indices][:, x_indices]
  uv_rows = uv[y_indices[::2] // 2]
  uv_small = np.stack((uv_rows[:, uv_x_indices], uv_rows[:, uv_x_indices + 1]), axis=-1)
  nv12 = np.concatenate((y_small.ravel(), uv_small.ravel()))
  return yolo_nv12_to_tensor(nv12, resized_w, resized_h, scale, dx, dy)


def set_yolo_cpu_affinity() -> tuple[int, ...] | None:
  """Keep the YOLO thread and ORT workers off modeld's realtime CPU."""
  if os.name != "posix" or not hasattr(os, "sched_setaffinity"):
    return None

  requested_cores = set(YOLO_CPU_CORES)
  # Use the Linux thread ID. PID 0 would change the affinity of the whole
  # modeld process, including the realtime model thread.
  thread_id = threading.get_native_id()
  try:
    os.sched_setaffinity(thread_id, requested_cores)
    return tuple(sorted(requested_cores))
  except OSError:
    available_cores = os.sched_getaffinity(thread_id)
    selected_cores = requested_cores & available_cores
    if not selected_cores:
      print(
        f"[road_yolo26n] CPU affinity skipped: requested={sorted(requested_cores)} "
        f"available={sorted(available_cores)}",
        flush=True,
      )
      return None
    os.sched_setaffinity(thread_id, selected_cores)
    return tuple(sorted(selected_cores))


def yolo_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
  if len(boxes) == 0:
    return []

  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 0] + boxes[:, 2]
  y2 = boxes[:, 1] + boxes[:, 3]
  areas = boxes[:, 2] * boxes[:, 3]
  order = scores.argsort()[::-1]
  keep = []

  while order.size > 0:
    i = int(order[0])
    keep.append(i)

    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
    order = order[np.where(iou <= threshold)[0] + 1]

  return keep


def yolo_detection_to_lead(detection: tuple[int, float, tuple[int, int, int, int]],
                           intrinsics: np.ndarray | None, intrinsics_size: tuple[int, int] | None,
                           img_w: int) -> YoloLead | None:
  if intrinsics is None or intrinsics_size is None:
    return None

  class_id, score, box = detection
  x, _, w, _ = box
  if w <= 0:
    return None

  real_width = YOLO_REAL_WIDTH_BY_CLASS[class_id] if 0 <= class_id < len(YOLO_REAL_WIDTH_BY_CLASS) else YOLO_DEFAULT_REAL_WIDTH

  intrinsics_w, _ = intrinsics_size
  focal = float(intrinsics[0, 0]) * (img_w / intrinsics_w)
  center = float(intrinsics[0, 2]) * (img_w / intrinsics_w)
  if focal <= 0.0:
    return None

  lead_x = (real_width * focal) / w
  center_x = x + w / 2.0
  lead_y = lead_x * ((center_x - center) / focal)
  return YoloLead(float(score), float(lead_x), float(lead_y), 0, 0, time.monotonic())


def yolo_lead_in_model_lane(model_output: dict[str, np.ndarray], yolo_lead: YoloLead) -> bool:
  lane_probs = model_output["lane_lines_prob"][0, 1::2]
  if len(lane_probs) < 2 or lane_probs[0] < YOLO_LANE_PROB_THRESHOLD or lane_probs[1] < YOLO_LANE_PROB_THRESHOLD:
    return False

  x_idxs = np.asarray(ModelConstants.X_IDXS, dtype=np.float32)
  left_lane_y = float(np.interp(yolo_lead.x, x_idxs, model_output["lane_lines"][0, 1, :, 0]))
  right_lane_y = float(np.interp(yolo_lead.x, x_idxs, model_output["lane_lines"][0, 2, :, 0]))
  lane_min = min(left_lane_y, right_lane_y) + YOLO_LANE_MARGIN
  lane_max = max(left_lane_y, right_lane_y) - YOLO_LANE_MARGIN
  return lane_min <= yolo_lead.y <= lane_max


def log_yolo_rejection(reason: str, has_yolo_lead: bool = True) -> None:
  global _last_yolo_rejection_reason, _last_yolo_rejection_log_t
  if not has_yolo_lead:
    return
  now = time.monotonic()
  if reason != _last_yolo_rejection_reason or now - _last_yolo_rejection_log_t >= 1.0:
    print(f"[road_yolo26n] modelV2 lead not replaced: {reason}", flush=True)
    _last_yolo_rejection_reason = reason
    _last_yolo_rejection_log_t = now


def apply_yolo_lead_if_needed(model_output: dict[str, np.ndarray], yolo_lead: YoloLead | None) -> bool:
  if not YOLO_ENABLED:
    log_yolo_rejection("YOLO disabled", has_yolo_lead=False)
    return False
  if yolo_lead is None:
    log_yolo_rejection("no YOLO lead", has_yolo_lead=False)
    return False
  lead_age = time.monotonic() - yolo_lead.created_t
  if lead_age > YOLO_LEAD_MAX_AGE:
    log_yolo_rejection(
      f"YOLO lead expired age={lead_age * 1000.0:.1f}ms "
      f"max={YOLO_LEAD_MAX_AGE * 1000.0:.1f}ms"
    )
    return False
  if not YOLO_USE_ONLY_LEAD and model_output["lead_prob"][0, 0] > 0.5:
    log_yolo_rejection(
      f"modelV2 lead has priority prob={float(model_output['lead_prob'][0, 0]):.3f}"
    )
    return False
  if not yolo_lead_in_model_lane(model_output, yolo_lead):
    log_yolo_rejection(
      f"YOLO lead outside model lane x={yolo_lead.x:.2f} y={yolo_lead.y:.2f}"
    )
    return False

  lead = np.zeros((ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float32)
  lead_stds = np.zeros_like(lead)
  model_v_ego = float(model_output["plan"][0, 0, Plan.VELOCITY][0])
  lead[:, 0] = yolo_lead.x
  lead[:, 1] = yolo_lead.y
  lead[:, 2] = model_v_ego
  lead[:, 3] = 0.0
  lead_stds[:, 0] = max(5.0, yolo_lead.x * 0.25)
  lead_stds[:, 1] = 2.0
  lead_stds[:, 2] = 10.0
  lead_stds[:, 3] = 5.0

  model_output["lead"][0] = lead
  model_output["lead_stds"][0] = lead_stds
  model_output["lead_prob"][0, 0] = yolo_lead.prob
  return True


class RoadYoloRunner:
  def __init__(self, model_path: Path = YOLO_MODEL_PATH, enabled: bool = YOLO_ENABLED,
               cl_context: CLContext | None = None):
    self.model_path = model_path
    self.enabled = enabled
    self.cl_context = cl_context
    self.thread: threading.Thread | None = None
    self.condition = threading.Condition()
    self.latest_lead: YoloLead | None = None
    self.latest_result_frame_id = -1
    self.completed_frame_id = -1
    self.pending_task: YoloTask | None = None
    self.active_preprocess_frame_id = -1
    self.expired_frame_id = -1
    self.failed = False
    self.use_opencl = YOLO_USE_OPENCL and cl_context is not None
    self.yolo_frame: YoloFrame | None = None
    self.preprocess_buffers: YoloPreprocessBuffers | None = None
    self.intrinsics: np.ndarray | None = None
    self.intrinsics_size: tuple[int, int] | None = None
    self.stats_frames = 0
    self.stats_elapsed_ms = 0.0
    self.stats_started_t = time.monotonic()
    self.ort_iobinding = None
    self.ort_output_buffers: list[np.ndarray] = []
    self.ort_output_names: list[str] = []
    self.postprocess_buffers: dict[str, np.ndarray] = {}
    self.postprocess_capacity = 0

  def start(self):
    if not self.enabled:
      cloudlog.warning("road yolo26n disabled by YOLO_ENABLED")
      return
    if self.thread is not None:
      return
    self.thread = threading.Thread(target=self._run, name="road_yolo26n", daemon=True)
    self.thread.start()

  def get_latest_lead(self) -> YoloLead | None:
    with self.condition:
      return self.latest_lead

  def submit(self, buf: VisionBuf, frame_id: int, timestamp_eof: int) -> bool:
    if not self.enabled:
      return False

    with self.condition:
      if self.failed:
        return False
      self.pending_task = YoloTask(frame_id, timestamp_eof, buf, time.perf_counter())
      self.condition.notify_all()
    return True

  def wait_for_frame(self, frame_id: int, timeout_ms: float = YOLO_WAIT_MS) -> YoloLead | None:
    if not self.enabled:
      return None

    deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
    with self.condition:
      while self.completed_frame_id < frame_id and not self.failed:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
          log_yolo_rejection(f"YOLO timeout wait_ms={timeout_ms:.1f}", has_yolo_lead=False)
          self.expired_frame_id = max(self.expired_frame_id, frame_id)
          if self.pending_task is not None and self.pending_task.frame_id <= self.expired_frame_id:
            self.pending_task = None
          while self.active_preprocess_frame_id == frame_id and not self.failed:
            self.condition.wait(timeout=0.001)
          self.condition.notify_all()
          return None
        self.condition.wait(timeout=remaining)
      if self.completed_frame_id == frame_id and self.latest_result_frame_id == frame_id:
        return self.latest_lead
      return None

  def set_camera_intrinsics(self, intrinsics: np.ndarray, width: int, height: int):
    if not self.enabled:
      return
    with self.condition:
      self.intrinsics = intrinsics.copy()
      self.intrinsics_size = (width, height)

  def _get_camera_intrinsics(self) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    with self.condition:
      return None if self.intrinsics is None else self.intrinsics.copy(), self.intrinsics_size

  def _set_latest_lead(self, lead: YoloLead | None, frame_id: int):
    with self.condition:
      if frame_id <= self.expired_frame_id:
        return
      self.latest_lead = lead
      self.latest_result_frame_id = frame_id
      self.completed_frame_id = max(self.completed_frame_id, frame_id)
      self.condition.notify_all()

  def _set_failed(self):
    with self.condition:
      self.failed = True
      self.condition.notify_all()

  def _postprocess(self, outputs: list[np.ndarray], scale: float, dx: int, dy: int,
                   img_w: int, img_h: int) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    pred = np.asarray(outputs[0])
    if pred.ndim == 3:
      pred = pred[0]
    elif pred.ndim != 2:
      raise ValueError(f"unsupported YOLO26 output shape: {pred.shape}")
    if pred.shape[0] == 84 and pred.shape[1] != 84:
      pred = pred.T
    elif pred.shape[1] != 84:
      raise ValueError(f"unsupported YOLO26 prediction shape: {pred.shape}")

    if pred.shape[1] <= 4:
      return []

    num_predictions = pred.shape[0]
    if self.postprocess_capacity < num_predictions or not self.postprocess_buffers:
      self.postprocess_capacity = num_predictions
      self.postprocess_buffers = {
        "rows": np.arange(num_predictions, dtype=np.intp),
        "class_ids": np.empty(num_predictions, dtype=np.intp),
        "scores": np.empty(num_predictions, dtype=np.float32),
        "valid": np.empty(num_predictions, dtype=bool),
        "normalized": np.empty(num_predictions, dtype=bool),
        "max_abs": np.empty(num_predictions, dtype=np.float32),
        "coords": np.empty((num_predictions, 4), dtype=np.float32),
        "x": np.empty(num_predictions, dtype=np.int32),
        "y": np.empty(num_predictions, dtype=np.int32),
        "bw": np.empty(num_predictions, dtype=np.int32),
        "bh": np.empty(num_predictions, dtype=np.int32),
        "x_float": np.empty(num_predictions, dtype=np.float32),
        "y_float": np.empty(num_predictions, dtype=np.float32),
        "bw_float": np.empty(num_predictions, dtype=np.float32),
        "bh_float": np.empty(num_predictions, dtype=np.float32),
        "valid_size": np.empty(num_predictions, dtype=bool),
        "compact_boxes": np.empty((num_predictions, 4), dtype=np.float32),
        "compact_scores": np.empty(num_predictions, dtype=np.float32),
        "compact_class_ids": np.empty(num_predictions, dtype=np.intp),
        "candidate_scores": np.empty(num_predictions, dtype=np.float32),
        "candidate_class_ids": np.empty(num_predictions, dtype=np.intp),
      }
      self.postprocess_buffers["obstacle_classes"] = np.zeros(pred.shape[1] - 4, dtype=bool)
      for class_id in YOLO_LANE_OBSTACLE_CLASS_IDS:
        if class_id < len(self.postprocess_buffers["obstacle_classes"]):
          self.postprocess_buffers["obstacle_classes"][class_id] = True

    buffers = self.postprocess_buffers
    class_scores = pred[:, 4:]
    class_ids = buffers["class_ids"][:num_predictions]
    scores = buffers["scores"][:num_predictions]
    valid = buffers["valid"][:num_predictions]
    np.copyto(class_ids, np.argmax(class_scores, axis=1))
    scores[:] = class_scores[buffers["rows"][:num_predictions], class_ids]
    np.greater_equal(scores, YOLO_CONF_THRESHOLD, out=valid)
    valid &= buffers["obstacle_classes"][class_ids]
    if not np.any(valid):
      return []

    valid_indices = np.flatnonzero(valid)
    valid_count = len(valid_indices)
    coords = buffers["coords"][:valid_count]
    candidate_scores = buffers["candidate_scores"][:valid_count]
    candidate_class_ids = buffers["candidate_class_ids"][:valid_count]
    np.take(pred[:, :4], valid_indices, axis=0, out=coords)
    np.take(scores, valid_indices, out=candidate_scores)
    np.take(class_ids, valid_indices, out=candidate_class_ids)
    normalized = buffers["normalized"][:valid_count]
    max_abs = buffers["max_abs"][:valid_count]
    np.max(np.abs(coords), axis=1, out=max_abs)
    np.less_equal(max_abs, 1.0, out=normalized)
    coords[normalized] *= np.array([YOLO_INPUT_W, YOLO_INPUT_H, YOLO_INPUT_W, YOLO_INPUT_H], dtype=np.float32)

    x = buffers["x"][:valid_count]
    y = buffers["y"][:valid_count]
    bw = buffers["bw"][:valid_count]
    bh = buffers["bh"][:valid_count]
    x_float = buffers["x_float"][:valid_count]
    y_float = buffers["y_float"][:valid_count]
    bw_float = buffers["bw_float"][:valid_count]
    bh_float = buffers["bh_float"][:valid_count]
    np.divide(coords[:, 0] - coords[:, 2] / 2.0 - dx, scale, out=x_float)
    np.divide(coords[:, 1] - coords[:, 3] / 2.0 - dy, scale, out=y_float)
    np.divide(coords[:, 2], scale, out=bw_float)
    np.divide(coords[:, 3], scale, out=bh_float)
    np.copyto(x, x_float, casting="unsafe")
    np.copyto(y, y_float, casting="unsafe")
    np.copyto(bw, bw_float, casting="unsafe")
    np.copyto(bh, bh_float, casting="unsafe")
    valid_size = buffers["valid_size"][:valid_count]
    np.greater(bw, 0, out=valid_size)
    valid_size &= bh > 0
    if not np.any(valid_size):
      return []

    valid_indices = np.flatnonzero(valid_size)
    valid_count = len(valid_indices)
    compact_boxes = buffers["compact_boxes"][:valid_count]
    compact_scores = buffers["compact_scores"][:valid_count]
    compact_class_ids = buffers["compact_class_ids"][:valid_count]
    compact_boxes[:, 0] = x[valid_indices]
    compact_boxes[:, 1] = y[valid_indices]
    compact_boxes[:, 2] = bw[valid_indices]
    compact_boxes[:, 3] = bh[valid_indices]
    np.clip(compact_boxes[:, 0], 0, img_w - 1, out=compact_boxes[:, 0])
    np.clip(compact_boxes[:, 1], 0, img_h - 1, out=compact_boxes[:, 1])
    np.minimum(compact_boxes[:, 2], img_w - compact_boxes[:, 0], out=compact_boxes[:, 2])
    np.minimum(compact_boxes[:, 3], img_h - compact_boxes[:, 1], out=compact_boxes[:, 3])
    np.maximum(compact_boxes[:, 2], 1, out=compact_boxes[:, 2])
    np.maximum(compact_boxes[:, 3], 1, out=compact_boxes[:, 3])
    np.take(candidate_scores, valid_indices, out=compact_scores)
    np.take(candidate_class_ids, valid_indices, out=compact_class_ids)

    keep = yolo_nms(compact_boxes, compact_scores, YOLO_NMS_THRESHOLD)
    return [
      (int(compact_class_ids[i]), round(float(compact_scores[i]), 6), tuple(int(value) for value in compact_boxes[i]))
      for i in keep
    ]

  def _preprocess_opencl(self, buf: VisionBuf) -> tuple[np.ndarray, float, int, int]:
    if self.yolo_frame is None:
      raise RuntimeError("YOLO OpenCL frame is not initialized")
    buffers = self._get_preprocess_buffers(buf)
    return buffers.process_opencl(self.yolo_frame, buf)

  def _preprocess_cpu(self, buf: VisionBuf) -> tuple[np.ndarray, float, int, int]:
    buffers = self._get_preprocess_buffers(buf)
    return buffers.process_cpu(buf)

  def _get_preprocess_buffers(self, buf: VisionBuf) -> YoloPreprocessBuffers:
    if self.preprocess_buffers is None or (
      self.preprocess_buffers.src_w != buf.width or self.preprocess_buffers.src_h != buf.height
    ):
      self.preprocess_buffers = YoloPreprocessBuffers(buf.width, buf.height)
    return self.preprocess_buffers

  def _run_inference(self, session, input_name: str, output_names: list[str],
                     input_tensor: np.ndarray) -> list[np.ndarray]:
    if self.ort_iobinding is None:
      return session.run(output_names, {input_name: input_tensor})

    try:
      self.ort_iobinding.clear_binding_inputs()
      self.ort_iobinding.bind_cpu_input(input_name, input_tensor)
      session.run_with_iobinding(self.ort_iobinding)
      return self.ort_output_buffers
    except Exception:
      # Keep the normal session.run path as a compatibility fallback for
      # older ONNX Runtime builds or dynamic output shapes.
      self.ort_iobinding = None
      self.ort_output_buffers = []
      return session.run(output_names, {input_name: input_tensor})

  def _run(self):
    try:
      yolo_cpu_core = set_yolo_cpu_affinity()
    except Exception:
      yolo_cpu_core = None
      cloudlog.exception("road yolo26n failed to set CPU affinity")

    try:
      import onnxruntime as ort
    except Exception:
      cloudlog.exception("road yolo26n disabled: failed to import onnxruntime")
      self._set_failed()
      return

    if not self.model_path.exists():
      cloudlog.warning(f"road yolo26n disabled: missing model at {self.model_path}")
      self._set_failed()
      return

    try:
      cloudlog.warning(
        f"[road_yolo26n] enabled=1 use_only_lead={int(YOLO_USE_ONLY_LEAD)} "
        f"preprocessing={'opencl' if self.use_opencl else 'cpu'} "
        f"cpu_cores={','.join(map(str, yolo_cpu_core)) if yolo_cpu_core is not None else 'default'} "
        f"wait_ms={YOLO_WAIT_MS:.1f}"
      )
      cloudlog.warning(f"road yolo26n preprocessing={'opencl' if self.use_opencl else 'cpu'}")
      session_options = ort.SessionOptions()
      session_options.intra_op_num_threads = YOLO_INTRA_OP_THREADS
      session_options.inter_op_num_threads = 1
      session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
      session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
      session_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
      session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
      input_name = session.get_inputs()[0].name
      output_names = [output.name for output in session.get_outputs()]
      self.ort_output_names = output_names
      try:
        output_infos = session.get_outputs()
        output_shapes = [tuple(int(dim) for dim in output.shape) for output in output_infos]
        if all(shape and all(dim > 0 for dim in shape) for shape in output_shapes):
          self.ort_output_buffers = [np.empty(shape, dtype=np.float32) for shape in output_shapes]
          self.ort_iobinding = session.io_binding()
          for output_name, output_buffer in zip(output_names, self.ort_output_buffers):
            self.ort_iobinding.bind_output(
              output_name, "cpu", 0, np.float32, output_buffer.shape, output_buffer.ctypes.data,
            )
      except Exception:
        self.ort_iobinding = None
        self.ort_output_buffers = []
    except Exception:
      cloudlog.exception("road yolo26n disabled: failed to load onnx model")
      self._set_failed()
      return

    cloudlog.warning("road yolo26n worker started")
    last_log_t = 0.0
    while True:
      with self.condition:
        while self.pending_task is None and not self.failed:
          self.condition.wait()
        if self.failed:
          return
        task = self.pending_task
        self.pending_task = None
        if task is None:
          continue
        if task.frame_id <= self.expired_frame_id:
          continue

      frame_start_t = time.perf_counter()
      lead = None
      detections = []
      preprocess_time_ms = 0.0
      inference_time_ms = 0.0
      postprocess_time_ms = 0.0
      try:
        with self.condition:
          self.active_preprocess_frame_id = task.frame_id
          self.condition.notify_all()

        preprocess_start_t = time.perf_counter()
        if self.use_opencl:
          try:
            if self.yolo_frame is None:
              self.yolo_frame = YoloFrame(self.cl_context)
            input_tensor, scale, dx, dy = self._preprocess_opencl(task.buf)
          except Exception:
            self.use_opencl = False
            self.yolo_frame = None
            cloudlog.exception("road yolo26n shared OpenCL preprocessing failed; falling back to CPU")
            input_tensor, scale, dx, dy = self._preprocess_cpu(task.buf)
        else:
          input_tensor, scale, dx, dy = self._preprocess_cpu(task.buf)
        preprocess_time_ms = (time.perf_counter() - preprocess_start_t) * 1000.0

        with self.condition:
          if self.active_preprocess_frame_id == task.frame_id:
            self.active_preprocess_frame_id = -1
          self.condition.notify_all()

        if task.frame_id <= self.expired_frame_id:
          continue

        inference_start_t = time.perf_counter()
        outputs = self._run_inference(session, input_name, output_names, input_tensor)
        inference_time_ms = (time.perf_counter() - inference_start_t) * 1000.0
        postprocess_start_t = time.perf_counter()
        detections = self._postprocess(outputs, scale, dx, dy, task.buf.width, task.buf.height)
        postprocess_time_ms = (time.perf_counter() - postprocess_start_t) * 1000.0
        intrinsics, intrinsics_size = self._get_camera_intrinsics()
        leads = [lead for detection in detections
                 if (lead := yolo_detection_to_lead(detection, intrinsics, intrinsics_size, task.buf.width)) is not None]
        lead = min(leads, key=lambda item: item.x) if leads else None
        if lead is not None:
          lead.frame_id = task.frame_id
          lead.timestamp_eof = task.timestamp_eof
      except Exception:
        cloudlog.exception("road yolo26n inference failed")
      finally:
        with self.condition:
          if self.active_preprocess_frame_id == task.frame_id:
            self.active_preprocess_frame_id = -1
            self.condition.notify_all()
        self._set_latest_lead(lead, task.frame_id)
        frame_time_ms = (time.perf_counter() - frame_start_t) * 1000.0
        self.stats_frames += 1
        self.stats_elapsed_ms += frame_time_ms
        if detections and frame_time_ms > YOLO_SLOW_FRAME_MS:
          print(
            f"[road_yolo26n] slow frame={task.frame_id} "
            f"elapsed={frame_time_ms:.2f}ms preprocess={preprocess_time_ms:.2f}ms "
            f"inference={inference_time_ms:.2f}ms postprocess={postprocess_time_ms:.2f}ms "
            f"detections={len(detections)}",
            flush=True,
          )

      now = time.monotonic()
      stats_elapsed_s = now - self.stats_started_t
      if lead is not None and stats_elapsed_s >= 1.0 and self.stats_frames > 0:
        avg_frame_ms = self.stats_elapsed_ms / self.stats_frames
        print(
          f"[road_yolo26n] average frames={self.stats_frames} "
          f"frame_time={avg_frame_ms:.2f}ms "
          f"fps={self.stats_frames / stats_elapsed_s:.2f} "
          f"use_only_lead={int(YOLO_USE_ONLY_LEAD)}",
          flush=True,
        )
        self.stats_frames = 0
        self.stats_elapsed_ms = 0.0
        self.stats_started_t = now
      if lead is not None and now - last_log_t > 1.0:
        lead_text = "none" if lead is None else f"x={lead.x:.1f} y={lead.y:.1f} prob={lead.prob:.2f}"
        cloudlog.info(f"road yolo26n frame={task.frame_id} detections={len(detections)} lead={lead_text}")
        last_log_t = now
