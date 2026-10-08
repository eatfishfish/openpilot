#!/usr/bin/env python3
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from cereal import messaging
from openpilot.common.basedir import BASEDIR


LOOP_TIMEOUT_MS = 100

LANE_CHANGE_MIN_SPEED_MS = 8.0
LANE_LINE_MIN_SPEED_MS = 8.0
LANE_LINE_PROB_THRESHOLD = 0.5
LANE_LINE_NEAR_MARGIN_M = 0.25

# Lead risk levels:
# notice: early reminder, no urgent action required yet.
# warning: pre-warning, driver should prepare to slow down.
# critical: warning, immediate attention required.
LEAD_NOTICE_HIGH_DELTA_DISTANCE_M = 130.0
LEAD_WARNING_HIGH_DELTA_DISTANCE_M = 90.0
LEAD_WARNING_TTC_S = 3.0
LEAD_CRITICAL_TTC_S = 1.5
LEAD_MIN_CLOSING_MS = 2.0
LEAD_MIN_DISTANCE_M = 8.0
LEAD_WARNING_HEADWAY_S = 1.2
LEAD_CRITICAL_HEADWAY_S = 0.6
LEAD_LOOKAHEAD_S = 2.0
LEAD_LOOKAHEAD_HEADWAY_S = 1.0
LEAD_HIGH_SPEED_MS = 22.0
LEAD_HIGH_DELTA_MS = 10.0

SOUND_COOLDOWN_S = 3.0
LEAD_NOTICE_SOUND = os.path.join(BASEDIR, "selfdrive/assets/sounds/prompt.wav")
LEAD_WARNING_SOUND = os.path.join(BASEDIR, "selfdrive/assets/sounds/slowdown.mp3")
LEAD_CRITICAL_SOUND = os.path.join(BASEDIR, "selfdrive/assets/sounds/warning_immediate.wav")
LANE_CHANGE_REFUSE_SOUND = os.path.join(BASEDIR, "selfdrive/assets/sounds/lane.mp3")
LINE_WARNING_SOUND = os.path.join(BASEDIR, "selfdrive/assets/sounds/line.mp3")


@dataclass(frozen=True)
class LeadAnalysis:
  status: bool = False
  distance_m: float = 0.0
  relative_speed_ms: float = 0.0
  lead_speed_ms: float = 0.0
  closing_speed_ms: float = 0.0
  ttc_s: Optional[float] = None
  predicted_distance_2s_m: float = 0.0
  safe_distance_m: float = 0.0
  critical_distance_m: float = 0.0
  level: str = "none"
  should_remind: bool = False
  should_warn: bool = False
  should_slow_down: bool = False
  reason: str = ""

  def as_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class LaneChangeAnalysis:
  left_allowed: bool = False
  right_allowed: bool = False
  desired_direction: str = "none"
  desired_allowed: bool = False
  reason: str = ""

  def as_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class LaneLineAnalysis:
  active: bool = False
  side: str = "none"
  distance_m: float = 0.0
  crossed: bool = False
  reason: str = ""

  def as_dict(self) -> Dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class FpAnalyzeInfo:
  timestamp: float
  v_ego_ms: float
  v_ego_kph: float
  left_blinker: bool
  right_blinker: bool
  left_blindspot: bool
  right_blindspot: bool
  lane_change: LaneChangeAnalysis
  lane_line: LaneLineAnalysis
  lead: LeadAnalysis

  def as_dict(self) -> Dict[str, Any]:
    return {
      "timestamp": self.timestamp,
      "v_ego_ms": self.v_ego_ms,
      "v_ego_kph": self.v_ego_kph,
      "left_blinker": self.left_blinker,
      "right_blinker": self.right_blinker,
      "left_blindspot": self.left_blindspot,
      "right_blindspot": self.right_blindspot,
      "lane_change": self.lane_change.as_dict(),
      "lane_line": self.lane_line.as_dict(),
      "lead": self.lead.as_dict(),
    }


class AsyncSoundPlayer:
  def __init__(self, cooldown_s: float = SOUND_COOLDOWN_S) -> None:
    self.cooldown_s = cooldown_s
    self._last_play_time: dict[str, float] = {}
    self._lock = threading.Lock()
    self._player = self._find_player()

  def play(self, sound_path: str, key: str) -> None:
    if not self._player or not os.path.exists(sound_path):
      return

    now = time.monotonic()
    with self._lock:
      if now - self._last_play_time.get(key, 0.0) < self.cooldown_s:
        return
      self._last_play_time[key] = now

    threading.Thread(target=self._play_blocking, args=(sound_path,), daemon=True).start()

  def _play_blocking(self, sound_path: str) -> None:
    try:
      command = [self._player, "-nodisp", "-autoexit", sound_path] if os.path.basename(self._player) == "ffplay" else [self._player, sound_path]
      subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
      )
    except OSError:
      pass

  @staticmethod
  def _find_player() -> Optional[str]:
    # try absolute paths first (PATH may be limited in openpilot env)
    candidates = [
      "/usr/bin/paplay",
      "/usr/bin/aplay",
      "/usr/bin/pw-play",
      "/usr/bin/ffplay",
      "paplay",
      "aplay",
      "pw-play",
      "ffplay",
    ]
    for player in candidates:
      if os.path.isfile(player) and os.access(player, os.X_OK):
        return player
      path = shutil.which(player)
      if path:
        return path
    return None


def analyze_lead(v_ego: float, lead) -> LeadAnalysis:
  if v_ego < 3 or not lead.status:
    return LeadAnalysis(reason="no_lead")

  d_rel = max(0.0, float(lead.dRel))
  v_rel = float(lead.vRel)
  v_lead = float(lead.vLead)
  closing_speed = max(0.0, -v_rel)
  ttc = d_rel / closing_speed if closing_speed > 0.1 else None
  warning_distance = max(LEAD_MIN_DISTANCE_M, v_ego * LEAD_WARNING_HEADWAY_S)
  critical_distance = max(LEAD_MIN_DISTANCE_M, v_ego * LEAD_CRITICAL_HEADWAY_S)
  lookahead_distance = max(0.0, d_rel - closing_speed * LEAD_LOOKAHEAD_S)
  lookahead_safe_distance = max(LEAD_MIN_DISTANCE_M, v_ego * LEAD_LOOKAHEAD_HEADWAY_S)

  level = "none"
  reason = "safe"

  if d_rel <= critical_distance and closing_speed >= LEAD_MIN_CLOSING_MS:
    level = "critical"
    reason = "critical_close_and_closing"
  elif ttc is not None and ttc <= LEAD_CRITICAL_TTC_S:
    level = "critical"
    reason = "critical_ttc"
  elif d_rel <= warning_distance and closing_speed >= LEAD_MIN_CLOSING_MS:
    level = "warning"
    reason = "close_and_closing"
  elif ttc is not None and ttc <= LEAD_WARNING_TTC_S:
    level = "warning"
    reason = "low_ttc"
  elif closing_speed >= LEAD_HIGH_DELTA_MS and lookahead_distance <= lookahead_safe_distance:
    level = "warning"
    reason = "unsafe_lookahead_gap"
  elif (
    v_ego >= LEAD_HIGH_SPEED_MS and
    closing_speed >= LEAD_HIGH_DELTA_MS and
    d_rel <= LEAD_WARNING_HIGH_DELTA_DISTANCE_M
  ):
    level = "warning"
    reason = "high_speed_large_delta"
  elif (
    v_ego >= LEAD_HIGH_SPEED_MS and
    closing_speed >= LEAD_HIGH_DELTA_MS and
    d_rel <= LEAD_NOTICE_HIGH_DELTA_DISTANCE_M
  ):
    level = "notice"
    reason = "notice_high_speed_large_delta"

  return LeadAnalysis(
    status=True,
    distance_m=d_rel,
    relative_speed_ms=v_rel,
    lead_speed_ms=v_lead,
    closing_speed_ms=closing_speed,
    ttc_s=ttc,
    predicted_distance_2s_m=lookahead_distance,
    safe_distance_m=warning_distance,
    critical_distance_m=critical_distance,
    level=level,
    should_remind=level in ("notice", "warning", "critical"),
    should_warn=level in ("warning", "critical"),
    should_slow_down=level in ("warning", "critical"),
    reason=reason,
  )


def analyze_lane_change(car_state) -> LaneChangeAnalysis:
  v_ego = float(car_state.vEgo)
  left_blinker = bool(car_state.leftBlinker)
  right_blinker = bool(car_state.rightBlinker)
  left_blindspot = bool(car_state.leftBlindspot)
  right_blindspot = bool(car_state.rightBlindspot)

  left_allowed = v_ego >= LANE_CHANGE_MIN_SPEED_MS and not left_blindspot
  right_allowed = v_ego >= LANE_CHANGE_MIN_SPEED_MS and not right_blindspot

  desired_direction = "none"
  desired_allowed = False
  reason = "no_blinker"
  if left_blinker != right_blinker:
    desired_direction = "left" if left_blinker else "right"
    desired_allowed = left_allowed if left_blinker else right_allowed
    if v_ego < LANE_CHANGE_MIN_SPEED_MS:
      reason = "too_slow"
    elif left_blinker and left_blindspot:
      reason = "left_blindspot"
    elif right_blinker and right_blindspot:
      reason = "right_blindspot"
    else:
      reason = "allowed"
  elif left_blinker and right_blinker:
    reason = "hazard_or_both_blinkers"

  return LaneChangeAnalysis(
    left_allowed=left_allowed,
    right_allowed=right_allowed,
    desired_direction=desired_direction,
    desired_allowed=desired_allowed,
    reason=reason,
  )


def analyze_lane_line(car_state, model_v2) -> LaneLineAnalysis:
  v_ego = float(car_state.vEgo)
  if v_ego < LANE_LINE_MIN_SPEED_MS:
    return LaneLineAnalysis(reason="too_slow")
  if bool(car_state.leftBlinker) or bool(car_state.rightBlinker):
    return LaneLineAnalysis(reason="blinker_active")
  if len(model_v2.laneLines) < 3 or len(model_v2.laneLineProbs) < 3:
    return LaneLineAnalysis(reason="no_lane_data")

  candidates: list[tuple[str, float, bool]] = []
  left_prob = float(model_v2.laneLineProbs[1])
  if left_prob >= LANE_LINE_PROB_THRESHOLD and len(model_v2.laneLines[1].y):
    left_y = float(model_v2.laneLines[1].y[0])
    candidates.append(("left", abs(left_y), left_y >= 0.0))

  right_prob = float(model_v2.laneLineProbs[2])
  if right_prob >= LANE_LINE_PROB_THRESHOLD and len(model_v2.laneLines[2].y):
    right_y = float(model_v2.laneLines[2].y[0])
    candidates.append(("right", abs(right_y), right_y <= 0.0))

  if not candidates:
    return LaneLineAnalysis(reason="lane_not_visible")

  side, distance_m, crossed = min(candidates, key=lambda item: item[1])
  if crossed:
    return LaneLineAnalysis(active=True, side=side, distance_m=distance_m, crossed=True, reason="line_crossed")
  if distance_m <= LANE_LINE_NEAR_MARGIN_M:
    return LaneLineAnalysis(active=True, side=side, distance_m=distance_m, crossed=False, reason="line_near")
  return LaneLineAnalysis(side=side, distance_m=distance_m, crossed=False, reason="safe")


def analyze(car_state, radar_state, model_v2) -> FpAnalyzeInfo:
  v_ego = max(0.0, float(car_state.vEgo))
  return FpAnalyzeInfo(
    timestamp=time.time(),
    v_ego_ms=v_ego,
    v_ego_kph=v_ego * 3.6,
    left_blinker=bool(car_state.leftBlinker),
    right_blinker=bool(car_state.rightBlinker),
    left_blindspot=bool(car_state.leftBlindspot),
    right_blindspot=bool(car_state.rightBlindspot),
    lane_change=analyze_lane_change(car_state),
    lane_line=analyze_lane_line(car_state, model_v2),
    lead=analyze_lead(v_ego, radar_state.leadOne),
  )


def play_lead_alert(sound_player: AsyncSoundPlayer, lead: LeadAnalysis) -> None:
  if lead.level == "critical":
    sound_player.play(LEAD_CRITICAL_SOUND, "lead_critical")
  elif lead.level == "warning":
    sound_player.play(LEAD_WARNING_SOUND, "lead_warning")
  elif lead.level == "notice":
    sound_player.play(LEAD_NOTICE_SOUND, "lead_notice")


def main() -> None:
  sound_player = AsyncSoundPlayer()
  sm = messaging.SubMaster(["carState", "radarState", "modelV2"], poll="carState")

  while True:
    sm.update(LOOP_TIMEOUT_MS)
    if not sm.updated["carState"]:
      continue

    info = analyze(sm["carState"], sm["radarState"], sm["modelV2"])
    play_lead_alert(sound_player, info.lead)

    if info.lane_line.active:
      sound_player.play(LINE_WARNING_SOUND, f"line_{info.lane_line.side}")
      print(
        f"line {info.lane_line.side}: d={info.lane_line.distance_m:.2f}m "
        f"crossed={info.lane_line.crossed} reason={info.lane_line.reason}"
      )

    if info.lead.level != "none":
      print(
        f"lead {info.lead.level}: d={info.lead.distance_m:.1f}m "
        f"vRel={info.lead.relative_speed_ms:.1f}m/s "
        f"ttc={info.lead.ttc_s if info.lead.ttc_s is not None else -1:.1f}s "
        f"reason={info.lead.reason}"
      )

    if info.lane_change.desired_direction != "none":
      print(
        f"lane_change {info.lane_change.desired_direction}: "
        f"{'allowed' if info.lane_change.desired_allowed else 'blocked'} "
        f"reason={info.lane_change.reason}"
      )
      if not info.lane_change.desired_allowed:
        sound_player.play(LANE_CHANGE_REFUSE_SOUND, f"lane_change_{info.lane_change.desired_direction}")


if __name__ == "__main__":
  main()
