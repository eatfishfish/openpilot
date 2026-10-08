#!/usr/bin/env python3
import math
from numbers import Number
import numpy as np

from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())

# 转弯减速参数：
# 如果进弯前速度还是过快，优先调低 TURN_SLOWDOWN_TARGET_LAT_ACCEL，例如 1.9 -> 1.6。
# 如果开始减速太晚，调大 TURN_SLOWDOWN_MAX_DISTANCE，例如 120.0 -> 150.0。
# 如果刹车力度不够，调低 TURN_SLOWDOWN_MIN_ACCEL，例如 -2.2 -> -2.8。
TURN_SLOWDOWN_MIN_SPEED = 8.0          # 低于此速度不触发转弯减速，单位 m/s。
TURN_SLOWDOWN_MIN_DISTANCE = 5.0       # 忽略太近的轨迹点，避免临近弯心时突然重刹。
TURN_SLOWDOWN_MAX_DISTANCE = 120.0     # 最远提前观察距离，越大越早为弯道准备减速。
TURN_SLOWDOWN_MIN_CURVATURE = 0.0025   # 最小弯道弧度，小于此值认为接近直道。
TURN_SLOWDOWN_TARGET_LAT_ACCEL = 2.0   # 目标过弯横向加速度，越小过弯目标速度越低。
TURN_SLOWDOWN_MIN_ACCEL = -1.5         # 转弯减速允许的最大减速度，越负刹车越强。
TURN_SLOWDOWN_DISTANCE_BUFFER = 1.0    # 距离缓冲，越大越保守，会更早完成减速。
TURN_SLOWDOWN_BRAKE_RATE = 1.8         # 转弯减速介入斜率，-2.8 m/s^2 约 1.6 秒建立完成。
TURN_SLOWDOWN_RELEASE_RATE = 0.9       # 转弯减速释放斜率，-2.8 m/s^2 约 3.1 秒平顺释放。


def turn_slowdown_accel(v_ego, model_v2):
  if v_ego < TURN_SLOWDOWN_MIN_SPEED:
    return 0.0

  positions = model_v2.position.x
  velocities = model_v2.velocity.x
  yaw_rates = model_v2.orientationRate.z
  if not (len(positions) and len(velocities) and len(yaw_rates)):
    return 0.0

  best_accel = 0.0
  for distance, velocity, yaw_rate in zip(positions, velocities, yaw_rates):
    distance = float(distance)
    if distance < TURN_SLOWDOWN_MIN_DISTANCE or distance > TURN_SLOWDOWN_MAX_DISTANCE:
      continue

    speed_at_point = max(float(velocity), 1.0)
    curvature = abs(float(yaw_rate)) / speed_at_point
    if curvature < TURN_SLOWDOWN_MIN_CURVATURE:
      continue

    target_speed = math.sqrt(TURN_SLOWDOWN_TARGET_LAT_ACCEL / curvature)
    if target_speed >= v_ego:
      continue

    braking_distance = max(distance - TURN_SLOWDOWN_DISTANCE_BUFFER, 1.0)
    required_accel = (target_speed ** 2 - v_ego ** 2) / (2.0 * braking_distance)
    best_accel = min(best_accel, required_accel)

  return float(np.clip(best_accel, TURN_SLOWDOWN_MIN_ACCEL, 0.0))


def smooth_turn_slowdown_accel(prev_accel, target_accel):
  if target_accel < prev_accel:
    max_delta = TURN_SLOWDOWN_BRAKE_RATE * DT_CTRL
  else:
    max_delta = TURN_SLOWDOWN_RELEASE_RATE * DT_CTRL

  return float(np.clip(target_accel, prev_accel - max_delta, prev_accel + max_delta))


def get_lane_offset_cm(params):
  try:
    return int(params.get("dp_lat_lane_offset", return_default=True))
  except (TypeError, ValueError):
    return -57


def lane_centering_curvature_offset(model_v2, v_ego, lane_offset_cm):
  lane_lines = model_v2.laneLines
  lane_line_probs = model_v2.laneLineProbs
  if len(lane_lines) < 3 or len(lane_line_probs) < 3:
    return 0.0

  # Check if left and right lane lines are detected with sufficient confidence.
  left_lane_visible = lane_line_probs[1] > 0.5
  right_lane_visible = lane_line_probs[2] > 0.5
  if not (left_lane_visible and right_lane_visible):
    return 0.0

  if not (len(lane_lines[1].y) and len(lane_lines[2].y) and len(model_v2.position.y)):
    return 0.0

  # Left lane Y is typically negative, right lane Y is typically positive.
  left_lane_y = lane_lines[1].y[0]
  right_lane_y = lane_lines[2].y[0]
  lane_center_y = (left_lane_y + right_lane_y) / 2.0
  user_offset_m = float(lane_offset_cm) / 100.0
  target_y = lane_center_y + user_offset_m

  safety_margin = 0.15
  target_y = np.clip(target_y, left_lane_y + safety_margin, right_lane_y - safety_margin)

  current_y = model_v2.position.y[0]
  y_error = target_y - current_y

  lookahead_dist = max(v_ego * 2.0, 10.0)
  if abs(lookahead_dist) <= 0.1:
    return 0.0

  return float((y_error / (lookahead_dist * lookahead_dist)) * 0.5)


class Controls:
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    self.CI = interfaces[self.CP.carFingerprint](self.CP)

    self.sm = messaging.SubMaster(['liveParameters', 'liveTorqueParameters', 'modelV2', 'selfdriveState',
                                   'liveCalibration', 'livePose', 'longitudinalPlan', 'carState', 'carOutput', 'radarState',
                                   'driverMonitoringState', 'onroadEvents', 'driverAssistance'], poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState', 'dpControlsState'])

    self.steer_limited_by_controls = False
    self.curvature = 0.0
    self.desired_curvature = 0.0

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    self.LoC = LongControl(self.CP)
    self.VM = VehicleModel(self.CP)
    self.LaC: LatControl
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      self.LaC = LatControlAngle(self.CP, self.CI)
    elif self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP, self.CI)
    elif self.CP.lateralTuning.which() == 'torque':
      self.LaC = LatControlTorque(self.CP, self.CI)

    self.alka_enabled = self.params.get_bool("dp_lat_alka")
    self.alka_active = False
    self.lane_offset_val = get_lane_offset_cm(self.params)
    self.prev_lane_offset_val = self.lane_offset_val
    self.turn_slowdown_accel = 0.0
    self.dp_lon_accel_resume = self.params.get_bool("dp_lon_accel_resume")
    self.accel_resume_request_frames = 0
    self.frame = 0

  def update(self):
    self.sm.update(15)
    self.frame += 1
    if self.frame % 100 == 0:
      self.lane_offset_val = get_lane_offset_cm(self.params)
      self.dp_lon_accel_resume = self.params.get_bool("dp_lon_accel_resume")
    if self.sm.updated["liveCalibration"]:
      self.pose_calibrator.feed_live_calib(self.sm['liveCalibration'])
    if self.sm.updated["livePose"]:
      device_pose = Pose.from_live_pose(self.sm['livePose'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)

  def state_control(self):
    CS = self.sm['carState']

    # Update VehicleModel
    lp = self.sm['liveParameters']
    x = max(lp.stiffnessFactor, 0.1)
    sr = max(lp.steerRatio, 0.1)
    self.VM.update_params(x, sr)

    steer_angle_without_offset = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
    self.curvature = -self.VM.calc_curvature(steer_angle_without_offset, CS.vEgo, lp.roll)

    # Update Torque Params
    if self.CP.lateralTuning.which() == 'torque':
      torque_params = self.sm['liveTorqueParameters']
      if self.sm.all_checks(['liveTorqueParameters']) and torque_params.useParams:
        self.LaC.update_live_torque_params(torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered,
                                           torque_params.frictionCoefficientFiltered)

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill
    self.alka_active = self.alka_enabled and CS.cruiseState.available and not standstill and CS.gearShifter != car.CarState.GearShifter.reverse
    lat_active = self.sm['selfdriveState'].active or self.alka_active
    CC.latActive = lat_active and not CS.steerFaultTemporary and not CS.steerFaultPermanent and \
                   (not standstill or self.CP.steerAtStandstill)
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in self.sm['onroadEvents']) and self.CP.openpilotLongitudinalControl

    actuators = CC.actuators
    actuators.longControlState = self.LoC.long_control_state

    # Enable blinkers while lane changing
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      CC.leftBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.left
      CC.rightBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.right

    if not CC.latActive:
      self.LaC.reset()
    if not CC.longActive:
      self.LoC.reset()

    # accel PID loop
    pid_accel_limits = self.CI.get_pid_accel_limits(self.CP, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    actuators.accel = float(self.LoC.update(CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits))
    if CC.longActive:
      raw_turn_accel = turn_slowdown_accel(CS.vEgo, model_v2)
      self.turn_slowdown_accel = smooth_turn_slowdown_accel(self.turn_slowdown_accel, raw_turn_accel)
      if self.turn_slowdown_accel < 0.0:
        actuators.accel = min(actuators.accel, self.turn_slowdown_accel)
    else:
      self.turn_slowdown_accel = 0.0

    if CC.latActive and CC.longActive:
    # Force braking if steer saturated
      if any(e.name.raw == log.OnroadEvent.EventName.steerSaturated for e in self.sm['onroadEvents']):
        actuators.accel = -5.0

      radarstate = self.sm['radarState']
      lead = radarstate.leadOne
      if lead is not None and lead.status and lead.fcw:
        close_obstacle_distance = np.interp(CS.vEgo, [0.0, 5.0, 10.0, 20.0], [3.0, 4.5, 7.0, 12.0])
        severe_radar_emergency = lead.radar and lead.dRel < close_obstacle_distance and lead.vRel < -5.0
        if severe_radar_emergency and CS.vEgo < 22.22:
          actuators.accel = -5.5

    curvature_offset = lane_centering_curvature_offset(model_v2, CS.vEgo, self.lane_offset_val) if CC.latActive else 0.0

    self.prev_lane_offset_val = self.lane_offset_val

    # Steering PID loop and lateral MPC
    # Reset desired curvature to current to avoid violating the limits on engage
    new_desired_curvature = model_v2.action.desiredCurvature if CC.latActive else self.curvature
    new_desired_curvature += curvature_offset
    self.desired_curvature, curvature_limited = clip_curvature(CS.vEgo, self.desired_curvature, new_desired_curvature, lp.roll)
    actuators.curvature = self.desired_curvature
    steer, steeringAngleDeg, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                                       self.steer_limited_by_controls, self.desired_curvature,
                                                       curvature_limited)  # TODO what if not available
    actuators.torque = float(steer)
    actuators.steeringAngleDeg = float(steeringAngleDeg)

    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)

    return CC, lac_log

  def publish(self, CC, lac_log):
    CS = self.sm['carState']

    # Orientation and angle rates can be useful for carcontroller
    # Only calibrated (car) frame is relevant for the carcontroller
    CC.currentCurvature = self.curvature
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    CC.cruiseControl.override = CC.enabled and not CC.longActive and self.CP.openpilotLongitudinalControl
    CC.cruiseControl.cancel = CS.cruiseState.enabled and (not CC.enabled or not self.CP.pcmCruise)
    accel_resume_event = any(e.name.raw == log.OnroadEvent.EventName.buttonEnable for e in self.sm['onroadEvents']) and \
                         any(e.name.raw == log.OnroadEvent.EventName.gasPressedOverride for e in self.sm['onroadEvents'])
    # The accelerator resume event is generated after braking has already
    # disengaged openpilot, so CC.enabled is expected to be false here.
    if self.dp_lon_accel_resume and accel_resume_event and not CS.cruiseState.enabled:
      self.accel_resume_request_frames = 5
    elif not self.dp_lon_accel_resume:
      self.accel_resume_request_frames = 0
    accel_resume_request = self.accel_resume_request_frames > 0
    if self.accel_resume_request_frames > 0:
      self.accel_resume_request_frames -= 1
    CC.cruiseControl.resume = (CC.enabled and CS.cruiseState.standstill and not self.sm['longitudinalPlan'].shouldStop) or accel_resume_request

    hudControl = CC.hudControl
    hudControl.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hudControl.speedVisible = CC.enabled
    hudControl.lanesVisible = CC.enabled
    hudControl.leadVisible = self.sm['longitudinalPlan'].hasLead
    hudControl.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    hudControl.visualAlert = self.sm['selfdriveState'].alertHudVisual

    hudControl.rightLaneVisible = True
    hudControl.leftLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hudControl.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hudControl.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    if self.sm['selfdriveState'].active:
      CO = self.sm['carOutput']
      if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
        self.steer_limited_by_controls = abs(CC.actuators.steeringAngleDeg - CO.actuatorsOutput.steeringAngleDeg) > \
                                              STEER_ANGLE_SATURATION_THRESHOLD
      else:
        self.steer_limited_by_controls = abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # dpControlsState
    dat = messaging.new_message('dpControlsState')
    dat.valid = True
    ncs = dat.dpControlsState
    ncs.alkaActive = self.alka_active
    self.pm.send('dpControlsState', dat)

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    cs.forceDecel = bool((self.sm['driverMonitoringState'].awarenessStatus < 0.) or
                         (self.sm['selfdriveState'].state == State.softDisabling))

    lat_tuning = self.CP.lateralTuning.which()
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      cs.lateralControlState.angleState = lac_log
    elif lat_tuning == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_tuning == 'torque':
      cs.lateralControlState.torqueState = lac_log

    self.pm.send('controlsState', dat)

    # carControl
    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    while True:
      self.update()
      CC, lac_log = self.state_control()
      self.publish(CC, lac_log)
      rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  controls = Controls()
  controls.run()


if __name__ == "__main__":
  main()
