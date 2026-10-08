import os
import re

from opendbc.car.can_definitions import CanData
from opendbc.car.structs import CarParams

SteerControlType = CarParams.SteerControlType
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
RESUME_BUTTON_PATH = os.path.join(BASEDIR, "resbtn.txt")
_RESUME_BUTTON_MSG = None
_RESUME_BUTTON_LOADED = False


def _load_resume_button_msg():
  global _RESUME_BUTTON_LOADED, _RESUME_BUTTON_MSG
  if _RESUME_BUTTON_LOADED:
    return _RESUME_BUTTON_MSG

  _RESUME_BUTTON_LOADED = True
  path = os.environ.get("TOYOTA_RESBTN_PATH", RESUME_BUTTON_PATH)
  try:
    with open(path, encoding="utf-8") as f:
      lines = f.readlines()
  except OSError:
    return None

  # Expected line format written by toyota_cam_can_dump.py after a RES/+ press:
  # selected bus=0 addr=0x365 dat=08000000fc0000
  msg_re = re.compile(r"\bbus=(\d+)\b.*\baddr=(0x[0-9a-fA-F]+|\d+)\b.*\bdat=([0-9a-fA-F]{2,16})\b")
  for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    if not line.startswith("selected "):
      continue
    match = msg_re.search(line)
    if match is None:
      continue

    bus = int(match.group(1), 0)
    address = int(match.group(2), 0)
    dat_hex = match.group(3)
    if len(dat_hex) % 2 != 0:
      continue
    _RESUME_BUTTON_MSG = CanData(address, bytes.fromhex(dat_hex), bus)
    return _RESUME_BUTTON_MSG

  return None


def create_steer_command(packer, steer, steer_req):
  """Creates a CAN message for the Toyota Steer Command."""

  values = {
    "STEER_REQUEST": steer_req,
    "STEER_TORQUE_CMD": steer,
    "SET_ME_1": 1,
  }
  return packer.make_can_msg("STEERING_LKA", 0, values)


def create_lta_steer_command(packer, steer_control_type, steer_angle, steer_req, frame, torque_wind_down):
  """Creates a CAN message for the Toyota LTA Steer Command."""

  values = {
    "COUNTER": frame + 128,
    "SETME_X1": 1,  # suspected LTA feature availability
    # 1 for TSS 2.5 cars, 3 for TSS 2.0. Send based on whether we're using LTA for lateral control
    "SETME_X3": 1 if steer_control_type == SteerControlType.angle else 3,
    "PERCENTAGE": 100,
    "TORQUE_WIND_DOWN": torque_wind_down,
    "ANGLE": 0,
    "STEER_ANGLE_CMD": steer_angle,
    "STEER_REQUEST": steer_req,
    "STEER_REQUEST_2": steer_req,
    "CLEAR_HOLD_STEERING_ALERT": 0,
  }
  return packer.make_can_msg("STEERING_LTA", 0, values)


def create_lta_steer_command_2(packer, frame):
  values = {
    "COUNTER": frame + 128,
  }
  return packer.make_can_msg("STEERING_LTA_2", 0, values)


def create_accel_command(packer, accel, pcm_cancel, permit_braking, standstill_req, lead, acc_type, fcw_alert, distance):
  # TODO: find the exact canceling bit that does not create a chime
  values = {
    "ACCEL_CMD": accel,
    "ACC_TYPE": acc_type,
    "DISTANCE": distance,
    "MINI_CAR": lead,
    "PERMIT_BRAKING": permit_braking,
    "RELEASE_STANDSTILL": not standstill_req,
    "CANCEL_REQ": pcm_cancel,
    "ALLOW_LONG_PRESS": 1,
    "ACC_CUT_IN": fcw_alert,  # only shown when ACC enabled
  }
  return packer.make_can_msg("ACC_CONTROL", 0, values)


def create_resume_button(packer):
  return _load_resume_button_msg()


def create_resume_button_test(packer):
  values = {
    "RES_BTN": 1,
    "SET_BTN": 0,
    "CANCEL_BTN": 0,
    "MAIN_ON": 0,
    "SET_SPEED": 0,
    "CRUISE_REQUEST": -12800,
    "LEAD_DISTANCE": 252,
  }
  return packer.make_can_msg("DSU_CRUISE", 0, values)

def create_pcs_commands(packer, accel, active, mass):
  values1 = {
    "COUNTER": 0,
    "FORCE": round(min(accel, 0) * mass * 2),
    "STATE": 3 if active else 0,
    "BRAKE_STATUS": 0,
    "PRECOLLISION_ACTIVE": 1 if active else 0,
  }
  msg1 = packer.make_can_msg("PRE_COLLISION", 0, values1)

  values2 = {
    "DSS1GDRV": min(accel, 0),     # accel
    "PCSALM": 1 if active else 0,  # goes high same time as PRECOLLISION_ACTIVE
    "IBTRGR": 1 if active else 0,  # unknown
    "PBATRGR": 1 if active else 0, # noisy actuation bit?
    "PREFILL": 1 if active else 0, # goes on and off before DSS1GDRV
    "AVSTRGR": 1 if active else 0,
  }
  msg2 = packer.make_can_msg("PRE_COLLISION_2", 0, values2)

  return [msg1, msg2]


def create_acc_cancel_command(packer):
  values = {
    "GAS_RELEASED": 0,
    "CRUISE_ACTIVE": 0,
    "ACC_BRAKING": 0,
    "ACCEL_NET": 0,
    "CRUISE_STATE": 0,
    "CANCEL_REQ": 1,
  }
  return packer.make_can_msg("PCM_CRUISE", 0, values)


def create_fcw_command(packer, fcw):
  values = {
    "PCS_INDICATOR": 1,  # PCS turned off
    "FCW": fcw,
    "SET_ME_X20": 0x20,
    "SET_ME_X10": 0x10,
    "PCS_OFF": 1,
    "PCS_SENSITIVITY": 0,
  }
  return packer.make_can_msg("PCS_HUD", 0, values)


def create_ui_command(packer, steer, chime, left_line, right_line, left_lane_depart, right_lane_depart, enabled, stock_lkas_hud):
  values = {
    "TWO_BEEPS": chime,
    "LDA_ALERT": steer,
    "RIGHT_LINE": 3 if right_lane_depart else 1 if right_line else 2,
    "LEFT_LINE": 3 if left_lane_depart else 1 if left_line else 2,
    "BARRIERS": 1 if enabled else 0,

    # static signals
    "SET_ME_X02": 2,
    "SET_ME_X01": 1,
    "LKAS_STATUS": 1,
    "REPEATED_BEEPS": 0,
    "LANE_SWAY_FLD": 7,
    "LANE_SWAY_BUZZER": 0,
    "LANE_SWAY_WARNING": 0,
    "LDA_FRONT_CAMERA_BLOCKED": 0,
    "TAKE_CONTROL": 0,
    "LANE_SWAY_SENSITIVITY": 2,
    "LANE_SWAY_TOGGLE": 1,
    "LDA_ON_MESSAGE": 0,
    "LDA_MESSAGES": 0,
    "LDA_SA_TOGGLE": 1,
    "LDA_SENSITIVITY": 2,
    "LDA_UNAVAILABLE": 0,
    "LDA_MALFUNCTION": 0,
    "LDA_UNAVAILABLE_QUIET": 0,
    "ADJUSTING_CAMERA": 0,
    "LDW_EXIST": 1,
  }

  # lane sway functionality
  # not all cars have LKAS_HUD — update with camera values if available
  if len(stock_lkas_hud):
    values.update({s: stock_lkas_hud[s] for s in [
      "LANE_SWAY_FLD",
      "LANE_SWAY_BUZZER",
      "LANE_SWAY_WARNING",
      "LANE_SWAY_SENSITIVITY",
      "LANE_SWAY_TOGGLE",
    ]})

  return packer.make_can_msg("LKAS_HUD", 0, values)


def toyota_checksum(address: int, sig, d: bytearray) -> int:
  s = len(d)
  addr = address
  while addr:
    s += addr & 0xFF
    addr >>= 8
  for i in range(len(d) - 1):
    s += d[i]
  return s & 0xFF
