#!/usr/bin/env python3
import argparse
from collections import deque
import os
import queue
import threading
import time

import cereal.messaging as messaging
from opendbc.can.dbc import DBC
from opendbc.can.parser import get_raw_value


DEFAULT_ADDRS = {
  0x191,  # STEERING_LTA
  0x283,  # PRE_COLLISION
  0x2E4,  # STEERING_LKA
  0x1D2,  # PCM_CRUISE
  0x1D3,  # PCM_CRUISE_2
  0x2E6,  # LEAD_INFO
  0x343,  # ACC_CONTROL
  0x365,  # DSU_CRUISE
  0x371,  # LTA_RELATED
  0x381,  # ACN1S04, reference DBC candidate: I_WIPD
  0x399,  # PCM_CRUISE_SM
  0x411,  # PCS_HUD
  0x412,  # LKAS_HUD
  0x614,  # BLINKERS_STATE
  0x622,  # LIGHT_STALK
}

DEFAULT_DBC = "toyota_nodsu_pt_generated"
ADAS_DBC = "toyota_tss2_adas"
TOYOTA_REF_DBC = "toyota_2017_ref_pt"
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_OUT = os.path.join(BASEDIR, "toyota_can_analysis.log")
DEFAULT_DISTANCE_OUT = os.path.join(BASEDIR, "distance.log")
DEFAULT_RESBTN_OUT = os.path.join(BASEDIR, "resbtn.txt")
DEFAULT_ROTATIONS = 5
DISTANCE_MAX_BYTES = 10 * 1024 * 1024
DISTANCE_ROTATIONS = 4
DISTANCE_LOG_INTERVAL = 0.1
TRACK_A_0_ADDR = 0x180
TRACK_A_0_SIGNALS = ("LONG_DIST", "LAT_DIST", "REL_SPEED", "VALID")
I_WIPD_ADDR = 0x381
I_WIPD_MSG = "ACN1S04"
I_WIPD_SIGNAL = "I_WIPD"
IS_WRITE_LOG = False
RES_TRIGGER_WINDOW = 0.35
RES_CANDIDATE_ADDRS = {
  0x1D2,  # PCM_CRUISE, CRUISE_STATE=9 when +/RES is pressed
  0x1D3,  # PCM_CRUISE_2, main/set speed/follow distance context
  0x343,  # ACC_CONTROL, if camera emits an ACC-side button effect
  0x365,  # DSU_CRUISE, legacy/compatibility RES_BTN if present
  0x399,  # PCM_CRUISE_SM, cluster cruise state context
}


def rotate_logs(path: str, rotations: int) -> None:
  if rotations <= 0:
    return
  oldest = f"{path}.{rotations}"
  if os.path.exists(oldest):
    os.remove(oldest)
  for idx in range(rotations - 1, 0, -1):
    src = f"{path}.{idx}"
    if os.path.exists(src):
      os.replace(src, f"{path}.{idx + 1}")
  if os.path.exists(path):
    os.replace(path, f"{path}.1")


def writer_thread(path: str, q: queue.Queue[str], max_bytes: int, rotations: int, flush_interval: float = 1.0) -> None:
  pending: list[str] = []
  pending_bytes = 0
  last_flush = time.monotonic()

  def flush_pending() -> None:
    nonlocal pending, pending_bytes, last_flush
    if not pending:
      last_flush = time.monotonic()
      return

    if IS_WRITE_LOG:
      try:
        if os.path.exists(path) and os.path.getsize(path) + pending_bytes > max_bytes:
          rotate_logs(path, rotations)
        with open(path, "a", buffering=1, encoding="utf-8") as f:
          f.writelines(pending)
      except OSError:
        pass
    pending = []
    pending_bytes = 0
    last_flush = time.monotonic()

  while True:
    try:
      line = q.get(timeout=flush_interval)
      pending.append(line)
      pending_bytes += len(line.encode("utf-8"))
    except queue.Empty:
      pass

    if time.monotonic() - last_flush >= flush_interval:
      flush_pending()


def parse_addr(addr: str) -> int:
  return int(addr, 0)


def decode_frame(dbc: DBC | None, address: int, dat: bytes) -> tuple[str, str, dict[str, float]]:
  if dbc is None:
    return "", "", {}

  msg = dbc.addr_to_msg.get(address)
  if msg is None:
    return "", "", {}

  values = []
  vals = {}
  for sig in msg.sigs.values():
    raw = get_raw_value(dat, sig)
    if sig.is_signed:
      raw -= ((raw >> (sig.size - 1)) & 0x1) * (1 << sig.size)
    val = raw * sig.factor + sig.offset
    vals[sig.name] = val
    values.append(f"{sig.name}={val:g}")
  return msg.name, f",{msg.name}:{';'.join(values)}", vals


def decode_track_a_0(dbc: DBC | None, address: int, dat: bytes) -> dict[str, float] | None:
  if dbc is None or address != TRACK_A_0_ADDR:
    return None

  msg = dbc.addr_to_msg.get(address)
  if msg is None or msg.name != "TRACK_A_0":
    return None

  vals = {}
  for sig_name in TRACK_A_0_SIGNALS:
    sig = msg.sigs.get(sig_name)
    if sig is None:
      return None

    raw = get_raw_value(dat, sig)
    if sig.is_signed:
      raw -= ((raw >> (sig.size - 1)) & 0x1) * (1 << sig.size)
    vals[sig_name] = raw * sig.factor + sig.offset

  return vals


def decode_i_wipd(dbc: DBC | None, address: int, dat: bytes) -> int | None:
  if dbc is None or address != I_WIPD_ADDR:
    return None

  msg = dbc.addr_to_msg.get(address)
  if msg is None or msg.name != I_WIPD_MSG:
    return None

  sig = msg.sigs.get(I_WIPD_SIGNAL)
  if sig is None:
    return None

  raw = get_raw_value(dat, sig)
  return int(raw)


def format_distance_line(wall_time: float, mono_time: int, bus: int, address: int, vals: dict[str, float]) -> str:
  return (
    f"{wall_time:.3f},{mono_time},bus={bus},addr=0x{address:X},TRACK_A_0:"
    f"LONG_DIST={vals['LONG_DIST']:.2f}m,"
    f"LAT_DIST={vals['LAT_DIST']:.2f}m,"
    f"REL_SPEED={vals['REL_SPEED']:.3f}m/s,"
    f"VALID={int(vals['VALID'])}\n"
  )


def changed_mask(prev: bytes | None, cur: bytes) -> str:
  if prev is None:
    return "ff" * len(cur)
  max_len = max(len(prev), len(cur))
  mask = []
  for i in range(max_len):
    old = prev[i] if i < len(prev) else 0
    new = cur[i] if i < len(cur) else 0
    mask.append("ff" if old != new else "00")
  return "".join(mask)


def describe_event(msg_name: str, vals: dict[str, float], prev_vals: dict[str, float] | None) -> str | None:
  if not prev_vals:
    return None

  def changed(sig: str) -> bool:
    return sig in vals and vals.get(sig) != prev_vals.get(sig)

  def rising(sig: str) -> bool:
    return sig in vals and vals.get(sig) == 1 and prev_vals.get(sig) != 1

  def falling(sig: str) -> bool:
    return sig in vals and vals.get(sig) == 0 and prev_vals.get(sig) != 0

  if msg_name == "PCM_CRUISE":
    if rising("CRUISE_ACTIVE"):
      return "ACC active"
    if falling("CRUISE_ACTIVE"):
      return "ACC inactive"
    if changed("CRUISE_STATE"):
      return f"ACC state changed to {vals['CRUISE_STATE']:g}"

  if msg_name == "PCM_CRUISE_2":
    if rising("MAIN_ON"):
      return "ACC main on"
    if falling("MAIN_ON"):
      return "ACC main off"
    if changed("SET_SPEED"):
      return f"ACC set speed changed to {vals['SET_SPEED']:g} kph"
    if rising("ACC_FAULTED"):
      return "ACC faulted"
    if changed("PCM_FOLLOW_DISTANCE"):
      return f"ACC follow distance changed to {vals['PCM_FOLLOW_DISTANCE']:g}"

  if msg_name == "ACC_CONTROL":
    if rising("CANCEL_REQ"):
      return "ACC cancel requested"
    if changed("ACCEL_CMD") and abs(vals["ACCEL_CMD"] - prev_vals.get("ACCEL_CMD", 0.0)) >= 0.5:
      return f"ACC accel command changed to {vals['ACCEL_CMD']:.3f} m/s^2"
    if rising("ACC_CUT_IN"):
      return "ACC cut-in warning"

  if msg_name == "BODY_CONTROL_STATE":
    doors = {
      "DOOR_OPEN_FL": "front left door",
      "DOOR_OPEN_FR": "front right door",
      "DOOR_OPEN_RL": "rear left door",
      "DOOR_OPEN_RR": "rear right door",
    }
    for sig, label in doors.items():
      if rising(sig):
        return f"{label} opened"
      if falling(sig):
        return f"{label} closed"
    if rising("SEATBELT_DRIVER_UNLATCHED"):
      return "driver seatbelt unlatched"
    if falling("SEATBELT_DRIVER_UNLATCHED"):
      return "driver seatbelt latched"
    if rising("PARKING_BRAKE"):
      return "parking brake on"
    if falling("PARKING_BRAKE"):
      return "parking brake off"

  if msg_name == "DOOR_LOCKS":
    if changed("LOCK_STATUS"):
      return "doors unlocked" if vals["LOCK_STATUS"] == 1 else "doors locked"
    if rising("LOCKED_VIA_KEYFOB"):
      return "locked via keyfob"
    if falling("LOCKED_VIA_KEYFOB"):
      return "keyfob lock released"

  if msg_name == "BLINKERS_STATE":
    if rising("HAZARD_LIGHT"):
      return "hazard lights on"
    if falling("HAZARD_LIGHT"):
      return "hazard lights off"
    if rising("BLINKER_BUTTON_PRESSED"):
      return "blinker/hazard button pressed"
    if not changed("TURN_SIGNALS"):
      return None
    turn_signals = int(vals["TURN_SIGNALS"])
    if turn_signals == 1:
      return "left turn signal"
    if turn_signals == 2:
      return "right turn signal"
    return "turn signal off"

  if msg_name == "PCS_HUD":
    if rising("FCW"):
      return "FCW warning"
    if falling("FCW"):
      return "FCW cleared"
    if rising("PCS_OFF"):
      return "PCS off"
    if rising("PCS_DUST") or rising("PCS_DUST2"):
      return "PCS camera dirty"
    if rising("PCS_TEMP") or rising("PCS_TEMP2"):
      return "PCS temperature unavailable"

  if msg_name == "PRE_COLLISION":
    if rising("PRECOLLISION_ACTIVE"):
      return f"pre-collision active force={vals.get('FORCE', 0.0):g}"
    if falling("PRECOLLISION_ACTIVE"):
      return "pre-collision inactive"

  if msg_name == "LKAS_HUD":
    if rising("LDA_ALERT"):
      return "LDA alert"
    if changed("LDA_ON_MESSAGE"):
      return f"LKA button/message changed to {vals['LDA_ON_MESSAGE']:g}"

  if msg_name == "LIGHT_STALK":
    light_signals = {
      "TAIL_LIGHT": "tail lights",
      "FRONT_FOG": "front fog lights",
      "PARKING_LIGHT": "parking lights",
      "LOW_BEAM": "low beams",
      "HIGH_BEAM": "high beams",
      "DAYTIME_RUNNING_LIGHT": "daytime running lights",
      "AUTO_HIGH_BEAM": "auto high beam",
    }
    for sig, label in light_signals.items():
      if rising(sig):
        return f"{label} on"
      if falling(sig):
        return f"{label} off"
    if rising("LIGHT_STALK_MOVED"):
      return "light stalk moved"
    if changed("AUTO_HIGH_BEAM"):
      return "auto high beam on" if vals["AUTO_HIGH_BEAM"] == 1 else "auto high beam off"
    if changed("HEADLIGHT_MODE"):
      return f"headlight mode changed to {vals['HEADLIGHT_MODE']:g}"

  return None


def is_resume_event(msg_name: str, vals: dict[str, float], prev_vals: dict[str, float] | None) -> bool:
  if not prev_vals:
    return False

  if msg_name == "DSU_CRUISE":
    return vals.get("RES_BTN") == 1 and prev_vals.get("RES_BTN") != 1

  # Toyota DBC documents PCM_CRUISE.CRUISE_STATE=9 as adaptive click up.
  # On Toyota steering wheels this is the RES/+ action.
  if msg_name == "PCM_CRUISE":
    return vals.get("CRUISE_STATE") == 9 and prev_vals.get("CRUISE_STATE") != 9

  return False


def format_can_line(prefix: str, wall_time: float, mono_time: int, bus: int, address: int,
                    dat: bytes, msg_name: str = "", decoded: str = "") -> str:
  name = f" msg={msg_name}" if msg_name else ""
  return f"{prefix} t={wall_time:.3f} mono={mono_time} bus={bus} addr=0x{address:X} dat={dat.hex()}{name}{decoded}\n"


def write_resume_capture(path: str, trigger_line: str, candidates: list[str]) -> None:
  selected_line = ""
  for line in candidates:
    if "msg=DSU_CRUISE" in line and "RES_BTN=1" in line:
      selected_line = "selected" + line[len("candidate"):]
      break

  try:
    with open(path, "w", encoding="utf-8") as f:
      f.write("# Toyota RES/+ button capture\n")
      f.write("# Confirm the selected line is the button command before testing replay.\n")
      f.write(trigger_line)
      if selected_line:
        f.write(selected_line)
      else:
        f.write("# No replayable selected frame was detected automatically.\n")
        f.write("# If one candidate is the real RES/+ command, copy it and change its prefix to selected.\n")
      for line in candidates:
        f.write(line)
      f.write("# create_resume_button loads only a line starting with: selected\n")
  except OSError as e:
    print(f"failed to write {path}: {e}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Safely dump and summarize Toyota CAN frames.")
  parser.add_argument("--bus", type=int, default=-1, help="-1 records all buses")
  parser.add_argument("--out", default=DEFAULT_OUT)
  parser.add_argument("--addr", type=parse_addr, nargs="*", default=None, help="optional address filter")
  parser.add_argument("--dbc", default=DEFAULT_DBC)
  parser.add_argument("--decode", action="store_true", default=True, help="append DBC-decoded signal values")
  parser.add_argument("--no-decode", dest="decode", action="store_false")
  parser.add_argument("--known-min-interval", type=float, default=0.5, help="minimum seconds between known DBC logs per bus/address")
  parser.add_argument("--unknown-min-interval", type=float, default=1.0, help="minimum seconds between unknown logs per bus/address")
  parser.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
  parser.add_argument("--rotations", type=int, default=DEFAULT_ROTATIONS)
  parser.add_argument("--distance-out", default=DEFAULT_DISTANCE_OUT)
  parser.add_argument("--queue-size", type=int, default=5000)
  parser.add_argument("--print", action="store_true", help="also print matching frames to console")
  parser.add_argument("--resbtn-out", default=DEFAULT_RESBTN_OUT)
  parser.add_argument("--capture-resume", action="store_true", default=True,
                      help="quietly capture a RES/+ press and write resbtn.txt")
  parser.add_argument("--no-capture-resume", dest="capture_resume", action="store_false")
  args = parser.parse_args()

  watch = set(args.addr) if args.addr is not None else None
  if watch is None:
    watch = DEFAULT_ADDRS
  dbc = DBC(args.dbc) if args.decode else None
  adas_dbc = DBC(ADAS_DBC)
  toyota_ref_dbc = DBC(TOYOTA_REF_DBC)
  q: queue.Queue[str] = queue.Queue(maxsize=args.queue_size)
  distance_q: queue.Queue[str] = queue.Queue(maxsize=args.queue_size)
  threading.Thread(target=writer_thread, args=(args.out, q, args.max_bytes, args.rotations), daemon=True).start()
  threading.Thread(
    target=writer_thread,
    args=(args.distance_out, distance_q, DISTANCE_MAX_BYTES, DISTANCE_ROTATIONS),
    daemon=True,
  ).start()

  dropped = 0
  stats: dict[tuple[int, int], dict[str, object]] = {}
  recent_frames: deque[tuple[float, int, int, int, bytes, str, str]] = deque(maxlen=300)
  resume_saved = False
  last_i_wipd_by_bus: dict[int, int] = {}
  last_distance_log_time = 0.0
  logcan = messaging.sub_sock("can")
  while True:
    for msg in messaging.drain_sock(logcan, wait_for_one=True):
      mono_time = msg.logMonoTime
      wall_time = time.time()
      for can in msg.can:
        if (args.bus >= 0 and can.src != args.bus) or (watch is not None and can.address not in watch):
          continue

        dat = bytes(can.dat)
        key = (can.src, can.address)
        track_a_0_vals = decode_track_a_0(adas_dbc, can.address, dat)
        if track_a_0_vals is not None and wall_time - last_distance_log_time >= DISTANCE_LOG_INTERVAL:
          distance_line = format_distance_line(wall_time, mono_time, can.src, can.address, track_a_0_vals)
          try:
            distance_q.put_nowait(distance_line)
          except queue.Full:
            dropped += 1
          last_distance_log_time = wall_time

        i_wipd = decode_i_wipd(toyota_ref_dbc, can.address, dat)
        if i_wipd is not None and last_i_wipd_by_bus.get(can.src) != i_wipd:
          last_i_wipd_by_bus[can.src] = i_wipd
          i_wipd_line = (
            f"{wall_time:.3f},{mono_time},bus={can.src},addr=0x{can.address:X},"
            f"{I_WIPD_MSG}:{I_WIPD_SIGNAL}={i_wipd},raw={dat.hex()}\n"
          )
          try:
            q.put_nowait(i_wipd_line)
          except queue.Full:
            dropped += 1
          if args.print and not args.capture_resume:
            print(i_wipd_line, end="")

        state = stats.setdefault(key, {
          "count": 0,
          "first_wall_time": wall_time,
          "last_log_time": 0.0,
          "last_dat": None,
          "last_decoded": "",
          "last_vals": None,
        })
        state["count"] = int(state["count"]) + 1

        msg_name, decoded, vals = decode_frame(dbc, can.address, dat)
        is_known = bool(decoded)
        if args.capture_resume:
          recent_frames.append((wall_time, mono_time, can.src, can.address, dat, msg_name, decoded))
        min_interval = args.known_min_interval if is_known else args.unknown_min_interval
        dat_changed = state["last_dat"] != dat
        decoded_changed = state["last_decoded"] != decoded
        should_log = (wall_time - float(state["last_log_time"]) >= min_interval) and (dat_changed or decoded_changed)
        event = describe_event(msg_name, vals, state["last_vals"]) if is_known else None
        resume_event = args.capture_resume and is_known and not resume_saved and \
                       is_resume_event(msg_name, vals, state["last_vals"])

        first_wall_time = float(state["first_wall_time"])
        freq = int(state["count"]) / max(wall_time - first_wall_time, 1e-3)
        mask = changed_mask(state["last_dat"], dat)
        status = "known" if is_known else "unknown"
        line = (
          f"{wall_time:.3f},{mono_time},bus={can.src},addr=0x{can.address:X},status={status},"
          f"count={state['count']},freq={freq:.1f},changed={mask},raw={dat.hex()}{decoded}\n"
        )
        if event is not None:
          event_line = f"{wall_time:.3f},{mono_time},bus={can.src},addr=0x{can.address:X},event={event}\n"
          try:
            q.put_nowait(event_line)
          except queue.Full:
            dropped += 1
          if args.print and not args.capture_resume:
            print(event_line, end="")

        if resume_event:
          trigger_line = format_can_line("trigger", wall_time, mono_time, can.src, can.address, dat, msg_name, decoded)
          candidates = []
          for f_wall_time, f_mono_time, f_bus, f_address, f_dat, f_msg_name, f_decoded in recent_frames:
            if wall_time - f_wall_time > RES_TRIGGER_WINDOW:
              continue
            if f_address not in RES_CANDIDATE_ADDRS:
              continue
            candidates.append(format_can_line("candidate", f_wall_time, f_mono_time, f_bus, f_address,
                                              f_dat, f_msg_name, f_decoded))
          write_resume_capture(args.resbtn_out, trigger_line, candidates)
          resume_saved = True
          print(f"saved RES/+ capture to {args.resbtn_out}")
          print(trigger_line, end="")
          for line in candidates:
            print(line, end="")

        if should_log:
          try:
            q.put_nowait(line)
          except queue.Full:
            dropped += 1

          if args.print and not args.capture_resume:
            print(line, end="")
          state["last_log_time"] = wall_time
          state["last_dat"] = dat
          state["last_decoded"] = decoded
        if is_known:
          state["last_vals"] = vals

    if dropped > 0 and q.qsize() < args.queue_size // 2:
      try:
        q.put_nowait(f"{time.time():.3f},0,-1,0x0,dropped={dropped}\n")
      except queue.Full:
        pass
      dropped = 0


if __name__ == "__main__":
  main()
