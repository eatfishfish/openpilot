#!/usr/bin/env python3
"""Manually test Toyota CAN commands over the sendcan bus.

Sends the same frames that carcontroller.py emits (door lock/unlock and
STEERING_LKA), referencing the pattern used in card.py:

    self.pm = messaging.PubMaster(['sendcan', ...])
    ...
    self.pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

Usage:
    python selfdrive/car/testcan.py setup [alloutput|toyota]  # 设置 panda safety 模式
    python selfdrive/car/testcan.py setup alloutput   # 放行一切,桌面/loopback 测试
    python selfdrive/car/testcan.py setup toyota      # 真实 toyota 模式(自动带 LOCK_CTRL 以便 0x750 放行)

    python selfdrive/car/testcan.py lock
    python selfdrive/car/testcan.py unlock

    python selfdrive/car/testcan.py steer [torque] [steer_req]
    python selfdrive/car/testcan.py steer 50 1     # 自定义
"""

import sys
import time

import cereal.messaging as messaging
from cereal import car
from openpilot.common.params import Params
from opendbc.can import CANPacker
from opendbc.car.can_definitions import CanData
from opendbc.car.toyota import toyotacan
from openpilot.selfdrive.pandad import can_list_to_can_capnp

# Lock / unlock door commands - Credit goes to AlexandreSato!
LOCK_UNLOCK_CAN_ID = 0x750
UNLOCK_CMD = b'\x40\x05\x30\x11\x00\x40\x00\x00'
LOCK_CMD = b'\x40\x05\x30\x11\x00\x80\x00\x00'

# default to the same CAN bus (0) the carcontroller sends on
BUS = 0
# Toyota powertrain DBC used by the carcontroller packer
DBC_NAME = 'toyota_nodsu_pt_generated'

# fixed small values for the steering test command
FIXED_APPLY_TORQUE = 100
FIXED_APPLY_STEER_REQ = 1

# Toyota LOCK_CTRL safetyParam flag (16 << 8), enables 0x750 lock/unlock tx
TOYOTA_PARAM_LOCK_CTRL = 16 << 8


def setup_safety(mode="alloutput"):
  """Put the panda into a given safety mode via the same params pandad reads.

  Based on selfdrive/pandad/tests/test_pandad_loopback.py::setup_pandad and
  selfdrive/pandad/panda_safety.cc::setSafetyMode. pandad only configures the
  safety model when IsOnroad/FirmwareQueryDone/ControlsReady are set and it
  reads CarParams.safetyConfigs.

  - alloutput: allow everything (desktop/loopback testing; no real actuation limits)
  - toyota:    real brand safety mode, subject to controls_allowed + torque checks.
               For 0x750 lock/unlock to pass, safetyParam must carry LOCK_CTRL.
  """
  params = Params()
  params.put_bool("IsOnroad", True)
  params.put_bool("FirmwareQueryDone", True)
  params.put_bool("ControlsReady", True)

  cp = car.CarParams.new_message()
  safety_config = car.CarParams.SafetyConfig.new_message()
  if mode == "toyota":
    safety_config.safetyModel = car.CarParams.SafetyModel.toyota
    # EPS scale byte 0 (100 default) + LOCK_CTRL so 0x750 is allowed
    safety_config.safetyParam = 100 | TOYOTA_PARAM_LOCK_CTRL
  else:
    safety_config.safetyModel = car.CarParams.SafetyModel.allOutput
  cp.safetyConfigs = [safety_config]

  params.put("CarParams", cp.to_bytes())

  # wait for pandad to apply the requested safety mode
  sm = messaging.SubMaster(['pandaStates'])
  target = car.CarParams.SafetyModel.allOutput if mode != "toyota" else car.CarParams.SafetyModel.toyota
  deadline = time.time() + 30
  while time.time() < deadline:
    sm.update(1000)
    if len(sm['pandaStates']) > 0 and all(ps.safetyModel == target for ps in sm['pandaStates']):
      print(f"panda safety mode set to {mode}")
      return
  print(f"WARNING: panda did not reach safety mode {mode} within 30s (last: {[ps.safetyModel for ps in sm['pandaStates']]})")


def send_can(msgs):
  # same sendcan publisher used by card.py (self.pm.sock['sendcan'])
  pm = messaging.PubMaster(['sendcan'])
  pm.send('sendcan', can_list_to_can_capnp(msgs, msgtype='sendcan', valid=True))

  # give pandad a moment to put the frames on the bus
  time.sleep(0.1)


def send_lock_unlock(action):
  cmd = LOCK_CMD if action == "lock" else UNLOCK_CMD
  send_can([CanData(LOCK_UNLOCK_CAN_ID, cmd, BUS)])
  print(f"sent {action.upper()}: addr=0x{LOCK_UNLOCK_CAN_ID:X} bus={BUS} dat={cmd.hex()}")


def send_steer(apply_torque, apply_steer_req):
  # same packer setup as toyota/carcontroller.py: self.packer = CANPacker(dbc_names[Bus.pt])
  packer = CANPacker(DBC_NAME)

  # same as carcontroller.py: steer_command = toyotacan.create_steer_command(self.packer, apply_torque, apply_steer_req)
  steer_command = toyotacan.create_steer_command(packer, apply_torque, apply_steer_req)

  # create_steer_command returns a (addr, bytes, bus) tuple accepted by can_list_to_can_capnp
  addr, dat, bus = steer_command
  send_can([steer_command])
  print(f"sent STEER: addr=0x{addr:X} bus={bus} torque={apply_torque} steer_req={apply_steer_req} dat={dat.hex()}")


def main():
  if len(sys.argv) < 2:
    print("Usage: python selfdrive/car/testcan.py <setup|lock|unlock|steer> ...")
    sys.exit(1)

  action = sys.argv[1]
  if action == "setup":
    mode = sys.argv[2] if len(sys.argv) > 2 else "alloutput"
    if mode not in ("alloutput", "toyota"):
      print("Usage: python selfdrive/car/testcan.py setup [alloutput|toyota]")
      sys.exit(1)
    setup_safety(mode)
  elif action in ("lock", "unlock"):
    send_lock_unlock(action)
  elif action == "steer":
    # default to fixed small values, allow override via args
    apply_torque = float(sys.argv[2]) if len(sys.argv) > 2 else FIXED_APPLY_TORQUE
    apply_steer_req = int(sys.argv[3]) if len(sys.argv) > 3 else FIXED_APPLY_STEER_REQ
    send_steer(apply_torque, apply_steer_req)
  else:
    print("Usage: python selfdrive/car/testcan.py <setup|lock|unlock|steer> ...")
    sys.exit(1)


if __name__ == "__main__":
  main()
