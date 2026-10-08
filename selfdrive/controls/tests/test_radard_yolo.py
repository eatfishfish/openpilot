from types import SimpleNamespace

from openpilot.selfdrive.controls.radard import get_lead


def make_lead(prob: float = 0.9, yolo_lead: bool = False, x: float = 20.0):
  return SimpleNamespace(
    prob=prob,
    x=[x],
    y=[0.0],
    v=[10.0],
    a=[0.0],
    xStd=[5.0],
    yStd=[2.0],
    vStd=[10.0],
    yoloLead=yolo_lead,
  )


def test_yolo_lead_does_not_fallback_to_visual_distance_without_radar():
  lead = get_lead(v_ego=10.0, ready=True, tracks={}, lead_msg=make_lead(yolo_lead=True),
                  model_v_ego=10.0)

  assert not lead["status"]


def test_non_yolo_lead_can_fallback_to_visual_distance():
  lead = get_lead(v_ego=10.0, ready=True, tracks={}, lead_msg=make_lead(),
                  model_v_ego=10.0)

  assert lead["status"]
  assert lead["radar"] is False
  assert lead["dRel"] == 20.0 - 1.52


def test_yolo_lead_uses_radar_distance_when_track_matches():
  radar_track = SimpleNamespace(
    cnt=3,
    dRel=18.5,
    yRel=0.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: False,
    get_RadarState=lambda model_prob: {
      "status": True,
      "radar": True,
      "dRel": 18.5,
      "modelProb": model_prob,
    },
  )

  lead = get_lead(v_ego=10.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True), model_v_ego=10.0)

  assert lead["status"]
  assert lead["radar"] is True
  assert lead["dRel"] == 18.5


def test_yolo_lead_allows_near_distance_error_up_to_five_meters():
  radar_track = SimpleNamespace(
    cnt=3,
    dRel=23.4,
    yRel=0.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: False,
    get_RadarState=lambda model_prob: {
      "status": True,
      "radar": True,
      "dRel": 23.4,
      "modelProb": model_prob,
    },
  )

  lead = get_lead(v_ego=10.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True, x=20.0), model_v_ego=10.0)

  assert lead["status"]
  assert lead["dRel"] == 23.4


def test_yolo_lead_allows_far_distance_error_up_to_thirty_five_percent():
  radar_track = SimpleNamespace(
    cnt=3,
    dRel=130.0,
    yRel=0.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: False,
    get_RadarState=lambda model_prob: {
      "status": True,
      "radar": True,
      "dRel": 130.0,
      "modelProb": model_prob,
    },
  )

  lead = get_lead(v_ego=10.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True, x=100.0), model_v_ego=10.0)

  assert lead["status"]
  assert lead["dRel"] == 130.0


def test_yolo_lead_rejects_distance_outside_dynamic_error():
  radar_track = SimpleNamespace(
    cnt=3,
    dRel=26.0,
    yRel=0.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: False,
    get_RadarState=lambda model_prob: {"status": True, "radar": True, "dRel": 26.0},
  )

  lead = get_lead(v_ego=10.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True, x=20.0), model_v_ego=10.0)

  assert not lead["status"]


def test_yolo_lead_rejects_unstable_radar_track():
  radar_track = SimpleNamespace(
    cnt=2,
    dRel=18.5,
    yRel=0.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: False,
    get_RadarState=lambda model_prob: {"status": True, "radar": True, "dRel": 18.5},
  )

  lead = get_lead(v_ego=10.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True), model_v_ego=10.0)

  assert not lead["status"]


def test_yolo_lead_rejects_radar_track_in_another_lateral_position():
  radar_track = SimpleNamespace(
    cnt=5,
    dRel=18.5,
    yRel=2.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: False,
    get_RadarState=lambda model_prob: {"status": True, "radar": True, "dRel": 18.5},
  )

  lead = get_lead(v_ego=10.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True), model_v_ego=10.0)

  assert not lead["status"]


def test_yolo_lead_does_not_use_unrelated_low_speed_radar_track():
  radar_track = SimpleNamespace(
    cnt=5,
    dRel=10.0,
    yRel=0.0,
    vRel=0.0,
    potential_low_speed_lead=lambda v_ego: True,
    get_RadarState=lambda model_prob=None: {
      "status": True,
      "radar": True,
      "dRel": 10.0,
      "modelProb": model_prob,
    },
  )

  lead = get_lead(v_ego=1.0, ready=True, tracks={1: radar_track},
                  lead_msg=make_lead(yolo_lead=True), model_v_ego=1.0)

  assert not lead["status"]
