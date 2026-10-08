#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in [ROOT, os.path.join(ROOT, "msgq_repo"), os.path.join(ROOT, "opendbc_repo")]:
  sys.path.insert(0, p)

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper


def main() -> None:
  pm = messaging.PubMaster(["pandaStates"])
  rk = Ratekeeper(10)

  while True:
    msg = messaging.new_message("pandaStates", 1)
    ps = msg.pandaStates[0]
    ps.pandaType = log.PandaState.PandaType.blackPanda
    ps.ignitionLine = True
    ps.ignitionCan = True
    ps.controlsAllowed = False
    ps.safetyModel = car.CarParams.SafetyModel.toyota
    ps.safetyParam = 100
    ps.faultStatus = log.PandaState.FaultStatus.none
    ps.harnessStatus = log.PandaState.HarnessStatus.normal
    ps.voltage = 12000
    ps.current = 0
    ps.uptime = rk.frame
    pm.send("pandaStates", msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
