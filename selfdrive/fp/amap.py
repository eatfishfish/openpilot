#!/usr/bin/env python3
import base64
import concurrent.futures
import datetime
import json
import math
import os
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha1
from typing import Any, Dict, Iterator, Optional

from openpilot.common.params import Params


AMAP_WS_PORT = 9888
CONNECT_TIMEOUT = 0.2
READ_TIMEOUT = 5.0
UNKNOWN_SPEED_LIMIT = 0
UNKNOWN_ROAD_SPEED_LIMIT = 20
SPEED_LIMIT_DROP_CONFIRMATIONS = 3
SPEED_LIMIT_LARGE_DROP_KPH = 40
AMAP_INPUT_COORD_SYS = os.getenv("AMAP_INPUT_COORD_SYS", "gcj02").lower()
GPS_INPUT_COORD_SYS = os.getenv("GPS_INPUT_COORD_SYS", "wgs84").lower()
AMAP_DISTANCE_STATS_PATH = os.getenv("AMAP_DISTANCE_STATS_PATH", "./amap_distance_stats.json")


def _env_float(name: str, default: float) -> float:
  try:
    return float(os.getenv(name, str(default)))
  except (TypeError, ValueError):
    return default


AMAP_WS_PULL_WINDOW_S = _env_float("AMAP_WS_PULL_WINDOW_S", 0.1)
AMAP_WS_EMPTY_RECONNECT_DELAY_S = _env_float("AMAP_WS_EMPTY_RECONNECT_DELAY_S", 0.2)
OPSERVER_GPS_MIN_DISTANCE_M = _env_float("OPSERVER_GPS_MIN_DISTANCE_M", 20.0)
OPSERVER_GPS_MIN_INTERVAL = _env_float("OPSERVER_GPS_MIN_INTERVAL", 1.0)
OPSERVER_GPS_TIMEOUT = _env_float("OPSERVER_GPS_TIMEOUT", 2.0)
AMAP_DISTANCE_MIN_SEGMENT_M = _env_float("AMAP_DISTANCE_MIN_SEGMENT_M", 3.0)
AMAP_DISTANCE_MAX_SPEED_KPH = _env_float("AMAP_DISTANCE_MAX_SPEED_KPH", 220.0)
AMAP_DISTANCE_MAX_INTERVAL_S = _env_float("AMAP_DISTANCE_MAX_INTERVAL_S", 5.0)
AMAP_DISTANCE_SAVE_INTERVAL_S = _env_float("AMAP_DISTANCE_SAVE_INTERVAL_S", 5.0)
AMAP_SERVER_SCAN_INTERVAL_S = _env_float("AMAP_SERVER_SCAN_INTERVAL_S", 1.0)


SPEED_LIMIT_RULES = (
  (("辅路", "匝道", "支路", "小区", "内部", "停车场", "服务区"), 30),
  (("高速", "高速公路"), 100),
  (("快速", "高架", "环线"), 80),
  (("大道", "大街", "主路", "主干"), 60),
  (("路", "街", "道"), 40),
)


@dataclass(frozen=True)
class AmapData:
  cur_road_name: str = ""
  next_road_name: str = ""
  cur_speed: int = 0
  limited_speed: int = 0
  unfiltered_limited_speed: int = 0
  raw_limited_speed: int = 0
  inferred_limited_speed: int = 0
  speed_limit_source: str = "unknown"
  speed_limit_corrected: bool = False
  speed_limit_correction_reason: str = ""
  seg_remain_dis: int = 0
  new_icon: int = 0
  turn_text: str = ""
  lane_num: int = 0
  amap_latitude: Optional[float] = None
  amap_longitude: Optional[float] = None
  gps_latitude: Optional[float] = None
  gps_longitude: Optional[float] = None
  gps_accuracy: Optional[float] = None
  gps_provider: str = ""
  gps_time: int = 0
  apk_publish_time_ms: int = 0
  amap_receive_time_ms: int = 0
  apk_to_amap_delay_ms: int = 0
  distance_day_m: float = 0.0
  distance_month_m: float = 0.0
  distance_year_m: float = 0.0
  distance_total_m: float = 0.0

  @property
  def road_name(self) -> str:
    return self.cur_road_name

  @property
  def speed_kph(self) -> int:
    return self.cur_speed

  @property
  def speed_limit_kph(self) -> int:
    return self.limited_speed

  @property
  def has_gps(self) -> bool:
    return self.gps_latitude is not None and self.gps_longitude is not None

  def as_dict(self) -> Dict[str, Any]:
    return asdict(self)

  def as_amap_dict(self) -> Dict[str, Any]:
    return {
      "CUR_ROAD_NAME": self.cur_road_name,
      "NEXT_ROAD_NAME": self.next_road_name,
      "CUR_SPEED": self.cur_speed,
      "LIMITED_SPEED": self.limited_speed,
      "UNFILTERED_LIMITED_SPEED": self.unfiltered_limited_speed,
      "RAW_LIMITED_SPEED": self.raw_limited_speed,
      "INFERRED_LIMITED_SPEED": self.inferred_limited_speed,
      "SPEED_LIMIT_SOURCE": self.speed_limit_source,
      "SPEED_LIMIT_CORRECTED": self.speed_limit_corrected,
      "SPEED_LIMIT_CORRECTION_REASON": self.speed_limit_correction_reason,
      "SEG_REMAIN_DIS": self.seg_remain_dis,
      "NEW_ICON": self.new_icon,
      "TURN_TEXT": self.turn_text,
      "LANE_NUM": self.lane_num,
      "AMAP_LATITUDE": self.amap_latitude,
      "AMAP_LONGITUDE": self.amap_longitude,
      "GPS_LATITUDE": self.gps_latitude,
      "GPS_LONGITUDE": self.gps_longitude,
      "GPS_ACCURACY": self.gps_accuracy,
      "GPS_PROVIDER": self.gps_provider,
      "GPS_TIME": self.gps_time,
      "APK_PUBLISH_TIME_MS": self.apk_publish_time_ms,
      "AMAP_RECEIVE_TIME_MS": self.amap_receive_time_ms,
      "APK_TO_AMAP_DELAY_MS": self.apk_to_amap_delay_ms,
      "DISTANCE_DAY_M": self.distance_day_m,
      "DISTANCE_MONTH_M": self.distance_month_m,
      "DISTANCE_YEAR_M": self.distance_year_m,
      "DISTANCE_TOTAL_M": self.distance_total_m,
    }


# GPS upload moved to FishMap APK. Keep amap.py focused on websocket parsing,
# distance stats, and Params updates so UI refresh is not blocked by network I/O.


class AmapDistanceStatsTracker:
  def __init__(
    self,
    path: str = AMAP_DISTANCE_STATS_PATH,
    min_segment_m: float = AMAP_DISTANCE_MIN_SEGMENT_M,
    max_speed_kph: float = AMAP_DISTANCE_MAX_SPEED_KPH,
    max_interval_s: float = AMAP_DISTANCE_MAX_INTERVAL_S,
    save_interval_s: float = AMAP_DISTANCE_SAVE_INTERVAL_S,
  ) -> None:
    self.path = path
    self.min_segment_m = min_segment_m
    self.max_speed_kph = max_speed_kph
    self.max_interval_s = max_interval_s
    self.save_interval_s = save_interval_s
    self._last_monotonic: Optional[float] = None
    self._last_point: Optional[tuple[float, float]] = None
    self._last_save_monotonic = 0.0
    self._stats = self._load()

  def update(self, data: AmapData) -> AmapData:
    now_mono = time.monotonic()
    now_wall = time.time()

    point = (data.gps_longitude, data.gps_latitude) if data.has_gps else None
    if self._last_monotonic is not None and self._last_point is not None and point is not None:
      dt = now_mono - self._last_monotonic
      distance_m = gps_distance_m(self._last_point, point)
      speed_kph = distance_m / dt * 3.6 if dt > 0.0 else 0.0
      if self.min_segment_m <= distance_m and 0.0 < dt <= self.max_interval_s and speed_kph <= self.max_speed_kph:
        self._add_distance(now_wall, distance_m)

    if point is not None:
      self._last_monotonic = now_mono
      self._last_point = point
    if now_mono - self._last_save_monotonic >= self.save_interval_s:
      self.save()
      self._last_save_monotonic = now_mono

    day_key, month_key, year_key = self._keys(now_wall)
    return replace(
      data,
      distance_day_m=self._stats["daily"].get(day_key, 0.0),
      distance_month_m=self._stats["monthly"].get(month_key, 0.0),
      distance_year_m=self._stats["yearly"].get(year_key, 0.0),
      distance_total_m=self._stats.get("total_m", 0.0),
    )

  def save(self) -> None:
    try:
      directory = os.path.dirname(self.path)
      if directory:
        os.makedirs(directory, exist_ok=True)
      tmp_path = f"{self.path}.tmp"
      with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(self._stats, f, ensure_ascii=False, sort_keys=True)
      os.replace(tmp_path, self.path)
    except OSError as exc:
      print(f"failed to save amap distance stats: {exc}")

  def get_stats(self) -> Dict[str, Any]:
    return dict(self._stats)

  def _add_distance(self, wall_time: float, distance_m: float) -> None:
    day_key, month_key, year_key = self._keys(wall_time)
    self._stats["daily"][day_key] = self._stats["daily"].get(day_key, 0.0) + distance_m
    self._stats["monthly"][month_key] = self._stats["monthly"].get(month_key, 0.0) + distance_m
    self._stats["yearly"][year_key] = self._stats["yearly"].get(year_key, 0.0) + distance_m
    self._stats["total_m"] = self._stats.get("total_m", 0.0) + distance_m

  def _load(self) -> Dict[str, Any]:
    try:
      with open(self.path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    except (OSError, ValueError):
      stats = {}

    return {
      "daily": dict(stats.get("daily", {}) if isinstance(stats.get("daily", {}), dict) else {}),
      "monthly": dict(stats.get("monthly", {}) if isinstance(stats.get("monthly", {}), dict) else {}),
      "yearly": dict(stats.get("yearly", {}) if isinstance(stats.get("yearly", {}), dict) else {}),
      "total_m": _as_float(stats.get("total_m")) or 0.0,
    }

  @staticmethod
  def _keys(wall_time: float) -> tuple[str, str, str]:
    date = datetime.datetime.fromtimestamp(wall_time).date()
    return date.isoformat(), date.strftime("%Y-%m"), date.strftime("%Y")


def gps_distance_m(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
  lon1, lat1 = point_a
  lon2, lat2 = point_b
  radius_m = 6371000.0
  phi1 = math.radians(lat1)
  phi2 = math.radians(lat2)
  delta_phi = math.radians(lat2 - lat1)
  delta_lambda = math.radians(lon2 - lon1)
  a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
  return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _as_int(value: Any, default: int = 0) -> int:
  if value is None:
    return default
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _as_float(value: Any) -> Optional[float]:
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def gcj02_to_wgs84(latitude: float, longitude: float) -> tuple[float, float]:
  if not (72.004 < longitude < 135.05 and 3.86 < latitude < 53.55):
    return latitude, longitude

  pi = math.pi
  axis = 6378245.0
  offset = 0.00669342162296594323

  x = longitude - 105.0
  y = latitude - 35.0
  d_lat = _transform_lat(x, y)
  d_lon = _transform_lon(x, y)
  rad_lat = latitude / 180.0 * pi
  magic = math.sin(rad_lat)
  magic = 1.0 - offset * magic * magic
  sqrt_magic = math.sqrt(magic)
  d_lat = (d_lat * 180.0) / ((axis * (1.0 - offset)) / (magic * sqrt_magic) * pi)
  d_lon = (d_lon * 180.0) / (axis / sqrt_magic * math.cos(rad_lat) * pi)
  return latitude * 2.0 - (latitude + d_lat), longitude * 2.0 - (longitude + d_lon)


def normalize_to_wgs84(latitude: float, longitude: float, coord_sys: str) -> tuple[float, float]:
  if coord_sys == "gcj02":
    return gcj02_to_wgs84(latitude, longitude)
  return latitude, longitude


def _transform_lat(x: float, y: float) -> float:
  ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
  ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y * math.pi / 3.0)) * 2.0 / 3.0
  ret += (160.0 * math.sin(y * math.pi / 12.0) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
  return ret


def _transform_lon(x: float, y: float) -> float:
  ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
  ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x * math.pi / 3.0)) * 2.0 / 3.0
  ret += (150.0 * math.sin(x * math.pi / 12.0) + 300.0 * math.sin(x * math.pi / 30.0)) * 2.0 / 3.0
  return ret


def infer_speed_limit(road_name: str) -> int:
  if not road_name or road_name in ("未知", "无名道路", "未知道路"):
    return UNKNOWN_ROAD_SPEED_LIMIT

  for keywords, speed_limit in SPEED_LIMIT_RULES:
    if any(keyword in road_name for keyword in keywords):
      return speed_limit
  return UNKNOWN_SPEED_LIMIT


def parse_amap_data(data: Dict[str, Any]) -> AmapData:
  receive_time_ms = int(time.time() * 1000)
  road_name = str(data.get("CUR_ROAD_NAME") or "")
  next_road_name = str(data.get("NEXT_ROAD_NAME") or "")
  raw_limited_speed = _as_int(data.get("LIMITED_SPEED"))
  inferred_limited_speed = infer_speed_limit(road_name) if raw_limited_speed <= 0 else 0
  limited_speed = raw_limited_speed if raw_limited_speed > 0 else inferred_limited_speed
  speed_limit_source = "amap" if raw_limited_speed > 0 else ("inferred" if inferred_limited_speed > 0 else "unknown")
  amap_latitude = _as_float(data.get("AMAP_LATITUDE"))
  amap_longitude = _as_float(data.get("AMAP_LONGITUDE"))
  gps_latitude = _as_float(data.get("GPS_LATITUDE"))
  gps_longitude = _as_float(data.get("GPS_LONGITUDE"))
  apk_publish_time_ms = _as_int(data.get("APK_PUBLISH_TIME_MS"))
  apk_to_amap_delay_ms = max(receive_time_ms - apk_publish_time_ms, 0) if apk_publish_time_ms > 0 else 0
  if gps_latitude is not None and gps_longitude is not None:
    gps_latitude, gps_longitude = normalize_to_wgs84(gps_latitude, gps_longitude, GPS_INPUT_COORD_SYS)
  elif amap_latitude is not None and amap_longitude is not None:
    gps_latitude, gps_longitude = normalize_to_wgs84(amap_latitude, amap_longitude, AMAP_INPUT_COORD_SYS)

  return AmapData(
    cur_road_name=road_name,
    next_road_name=next_road_name,
    cur_speed=_as_int(data.get("CUR_SPEED")),
    limited_speed=limited_speed,
    unfiltered_limited_speed=limited_speed,
    raw_limited_speed=raw_limited_speed,
    inferred_limited_speed=inferred_limited_speed,
    speed_limit_source=speed_limit_source,
    seg_remain_dis=_as_int(data.get("SEG_REMAIN_DIS")),
    new_icon=_as_int(data.get("NEW_ICON")),
    turn_text=str(data.get("TURN_TEXT") or ""),
    lane_num=_as_int(data.get("LANE_NUM")),
    amap_latitude=amap_latitude,
    amap_longitude=amap_longitude,
    gps_latitude=gps_latitude,
    gps_longitude=gps_longitude,
    gps_accuracy=_as_float(data.get("GPS_ACCURACY")),
    gps_provider=str(data.get("GPS_PROVIDER") or ""),
    gps_time=_as_int(data.get("GPS_TIME")),
    apk_publish_time_ms=apk_publish_time_ms,
    amap_receive_time_ms=receive_time_ms,
    apk_to_amap_delay_ms=apk_to_amap_delay_ms,
  )


class AmapSpeedLimitSmoother:
  def __init__(
    self,
    drop_confirmations: int = SPEED_LIMIT_DROP_CONFIRMATIONS,
    large_drop_kph: int = SPEED_LIMIT_LARGE_DROP_KPH,
  ) -> None:
    self.drop_confirmations = drop_confirmations
    self.large_drop_kph = large_drop_kph
    self._stable_limit = UNKNOWN_SPEED_LIMIT
    self._pending_lower_limit = UNKNOWN_SPEED_LIMIT
    self._pending_lower_count = 0

  def update(self, data: AmapData) -> AmapData:
    candidate = data.limited_speed
    if candidate <= 0:
      return data

    if self._stable_limit <= 0:
      self._accept(candidate)
      return data

    if candidate >= self._stable_limit:
      self._accept(candidate)
      return data

    drop = self._stable_limit - candidate
    if drop < self.large_drop_kph:
      self._accept(candidate)
      return data

    if candidate != self._pending_lower_limit:
      self._pending_lower_limit = candidate
      self._pending_lower_count = 1
    else:
      self._pending_lower_count += 1

    if self._pending_lower_count >= self.drop_confirmations:
      self._accept(candidate)
      return data

    return replace(
      data,
      limited_speed=self._stable_limit,
      speed_limit_source=f"{data.speed_limit_source}_corrected",
      speed_limit_corrected=True,
      speed_limit_correction_reason=(
        f"suppressed sudden drop {self._stable_limit}->{candidate} "
        f"({self._pending_lower_count}/{self.drop_confirmations})"
      ),
    )

  def _accept(self, speed_limit: int) -> None:
    self._stable_limit = speed_limit
    self._pending_lower_limit = UNKNOWN_SPEED_LIMIT
    self._pending_lower_count = 0


def get_lan_ip():
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    try:
      sock.connect(("8.8.8.8", 80))
      return sock.getsockname()[0]
    except OSError:
      try:
        return socket.gethostbyname(socket.gethostname())
      except OSError:
        return "127.0.0.1"


def iter_subnet_hosts(local_ip):
  parts = local_ip.split(".")
  if len(parts) != 4:
    return

  prefix = ".".join(parts[:3])
  try:
    local_last = int(parts[3])
  except (TypeError, ValueError):
    return
  for last in range(1, 255):
    if last == local_last:
      continue
    yield f"{prefix}.{last}"


def _env_server_hosts() -> list[str]:
  raw = os.getenv("AMAP_SERVER_HOST") or os.getenv("FISHMAP_SERVER_HOST") or ""
  hosts = []
  for value in raw.replace(";", ",").split(","):
    host = value.strip()
    if host:
      hosts.append(host)
  return hosts


def websocket_connect(host, port=AMAP_WS_PORT):
  key = base64.b64encode(os.urandom(16)).decode("ascii")
  request = (
    f"GET / HTTP/1.1\r\n"
    f"Host: {host}:{port}\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    "Sec-WebSocket-Version: 13\r\n\r\n"
  )

  sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
  sock.sendall(request.encode("ascii"))
  response = sock.recv(4096).decode("iso-8859-1", errors="replace")
  expected_accept = base64.b64encode(
    sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
  ).decode("ascii")

  if " 101 " not in response or expected_accept not in response:
    sock.close()
    raise ConnectionError("invalid websocket handshake")

  sock.settimeout(READ_TIMEOUT)
  return sock


def _try_connect_amap(host: str) -> tuple[str, socket.socket] | None:
  try:
    return host, websocket_connect(host)
  except (OSError, ConnectionError):
    return None


def find_amap_server(local_ip=None):
  local_ip = local_ip or get_lan_ip()
  explicit_hosts = _env_server_hosts()
  hosts = []
  if explicit_hosts:
    hosts.extend(explicit_hosts)
  hosts.extend(iter_subnet_hosts(local_ip) or [])
  # print("hosts = ", hosts)
  with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = {executor.submit(_try_connect_amap, host): host for host in hosts}
    for future in concurrent.futures.as_completed(futures):
      try:
        result = future.result()
        if result is not None:
          # cancel remaining scans, early return
          for f in futures:
            f.cancel()
          return result
      except Exception:
        continue

  if explicit_hosts:
    raise ConnectionError(f"no FishMap websocket server reachable at configured hosts: {', '.join(explicit_hosts)}")
  raise ConnectionError(f"no FishMap websocket server found near {local_ip}")


def recv_exact(sock, size):
  chunks = bytearray()
  while len(chunks) < size:
    chunk = sock.recv(size - len(chunks))
    if not chunk:
      raise ConnectionError("websocket closed")
    chunks.extend(chunk)
  return bytes(chunks)


def recv_ws_text(sock):
  header = recv_exact(sock, 2)
  opcode = header[0] & 0x0f
  masked = (header[1] & 0x80) != 0
  length = header[1] & 0x7f

  if length == 126:
    length = struct.unpack("!H", recv_exact(sock, 2))[0]
  elif length == 127:
    length = struct.unpack("!Q", recv_exact(sock, 8))[0]

  mask = recv_exact(sock, 4) if masked else None
  payload = bytearray(recv_exact(sock, length))
  if mask:
    for i in range(length):
      payload[i] ^= mask[i % 4]

  if opcode == 0x8:
    raise ConnectionError("websocket close frame")
  if opcode != 0x1:
    return None

  return payload.decode("utf-8")


def receive_amap_data() -> Iterator[Dict[str, Any]]:
  next_scan_time = 0.0
  while True:
    now = time.monotonic()
    if now < next_scan_time:
      time.sleep(next_scan_time - now)
    next_scan_time = time.monotonic() + AMAP_SERVER_SCAN_INTERVAL_S

    local_ip = get_lan_ip()
    print(f"local ip: {local_ip}, scanning {'.'.join(local_ip.split('.')[:3])}.x:{AMAP_WS_PORT}")
    try:
      host, sock = find_amap_server(local_ip)
      print(f"connected to FishMap websocket: ws://{host}:{AMAP_WS_PORT}")
      with sock:
        sock.settimeout(min(READ_TIMEOUT, AMAP_WS_PULL_WINDOW_S))
        deadline = time.monotonic() + AMAP_WS_PULL_WINDOW_S
        received = False
        while time.monotonic() < deadline:
          try:
            message = recv_ws_text(sock)
          except socket.timeout:
            break
          if not message:
            continue
          received = True
          try:
            data = json.loads(message)
          except (TypeError, json.JSONDecodeError, ValueError) as exc:
            print(f"invalid amap json ignored: {exc}")
            continue
          if not isinstance(data, dict):
            print(f"invalid amap payload ignored: {type(data).__name__}")
            continue
          yield data
      if not received:
        time.sleep(AMAP_WS_EMPTY_RECONNECT_DELAY_S)
    except Exception as exc:
      print(f"amap websocket disconnected: {exc}")
      time.sleep(1.0)


def receive_parsed_amap_data() -> Iterator[AmapData]:
  smoother = AmapSpeedLimitSmoother()
  for data in receive_amap_data():
    try:
      yield smoother.update(parse_amap_data(data))
    except Exception as exc:
      print(f"invalid amap data ignored: {exc}")


def smooth_amap_data(data_stream: Iterator[AmapData]) -> Iterator[AmapData]:
  smoother = AmapSpeedLimitSmoother()
  for data in data_stream:
    yield smoother.update(data)


def main():
  # GPS upload moved to FishMap APK.
  # gps_uploader = GpsUploader()
  distance_tracker = AmapDistanceStatsTracker()
  params = Params()

  def _run() -> None:
    for data in receive_parsed_amap_data():
      try:
        data = distance_tracker.update(data)
        # GPS upload moved to FishMap APK so websocket/Params updates are never blocked here.
        # gps_uploader.maybe_upload(data)
        amap_dict = data.as_amap_dict()
        amap_json = json.dumps(amap_dict, ensure_ascii=False, separators=(",", ":"))
        # Params expects JSON-typed params to be written as Python dict/list,
        # not as a JSON string. Write the dict directly and keep printing
        # the JSON string for logging.
        params.put_nonblocking("AmapInfo", amap_dict)
        print(amap_json)
      except Exception as exc:
        print(f"failed to publish amap data: {exc}")

  t = threading.Thread(target=_run, daemon=True, name="amap")
  t.start()
  # keep process alive until the daemon thread exits or process is killed
  t.join()


if __name__ == "__main__":
  main()
