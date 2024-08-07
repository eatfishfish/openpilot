#!/usr/bin/env python3
import json
import math
import os
import threading

import requests
import time  ###Upload GPS###

from openpilot.common.numpy_fast import interp

import cereal.messaging as messaging
from cereal import log
from openpilot.common.api import Api
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.navd.helpers import (Coordinate, coordinate_from_param,
                                    distance_along_geometry, maxspeed_to_ms,
                                    minimum_distance,
                                    parse_banner_instructions)
from openpilot.common.swaglog import cloudlog

REROUTE_DISTANCE = 25
MANEUVER_TRANSITION_THRESHOLD = 10
REROUTE_COUNTER_MIN = 3


class RouteEngine:
  def __init__(self, sm, pm):
    self.sm = sm
    self.pm = pm

    self.params = Params()

    # Get last gps position from params
    self.last_position = coordinate_from_param("LastGPSPosition", self.params)
    self.last_bearing = None

    self.gps_ok = False
    self.localizer_valid = False

    self.nav_destination = None
    self.step_idx = None
    self.route = None
    self.route_geometry = None

    self.recompute_backoff = 0
    self.recompute_countdown = 0

    self.ui_pid = None

    self.reroute_counter = 0

    self.last_upload_time = 0   ###Upload GPS### last upload time
    self.last_upload_gps = self.last_position   ###Upload GPS### last upload gps

    self.api = None
    self.mapbox_token = None
    if "MAPBOX_TOKEN" in os.environ:
      self.mapbox_token = os.environ["MAPBOX_TOKEN"]
      self.mapbox_host = "https://api.mapbox.com"
    else:
      self.api = Api(self.params.get("DongleId", encoding='utf8'))
      self.mapbox_host = "https://maps.comma.ai"
    self.mapbox_host = "http://laofolan.tpddns.cn:9988"


  def update(self):
    self.sm.update(0)

    if self.sm.updated["managerState"]:
      ui_pid = [p.pid for p in self.sm["managerState"].processes if p.name == "ui" and p.running]
      if ui_pid:
        if self.ui_pid and self.ui_pid != ui_pid[0]:
          cloudlog.warning("UI restarting, sending route")
          threading.Timer(5.0, self.send_route).start()
        self.ui_pid = ui_pid[0]

    self.update_location()
    try:
      self.recompute_route()
      self.send_instruction()
    except Exception:
      cloudlog.exception("navd.failed_to_compute")

  def update_location(self):
    location = self.sm['liveLocationKalman']
    self.gps_ok = location.gpsOK

    self.localizer_valid = (location.status == log.LiveLocationKalman.Status.valid) and location.positionGeodetic.valid

    if self.localizer_valid:
      self.last_bearing = math.degrees(location.calibratedOrientationNED.value[2])
      self.last_position = Coordinate(location.positionGeodetic.value[0], location.positionGeodetic.value[1])

    ###Upload GPS###
    timestamp = int(time.time())
    if self.gps_ok:
      offLng = self.last_upload_gps.longitude - self.last_position.longitude
      offLat = self.last_upload_gps.latitude - self.last_position.latitude

      time_interval = 10
      if (offLng < 0.0000001 and offLng > -0.0000001) and (offLat < 0.0000001 and offLat > -0.0000001):
        time_interval = 60

      if timestamp - self.last_upload_time > time_interval:
        self.last_upload_time = timestamp
        API_HOST = os.getenv('API_HOST', '')
        if API_HOST == '':
          return

        dongle_id = Params().get("DongleId", encoding='utf-8')
        url = API_HOST+ "/gps/"+dongle_id+"/"+str(self.last_position.longitude)+","+str(self.last_position.latitude)
        try:
          requests.put(url,timeout=2)
          self.last_upload_gps = self.last_position
        except Exception:
          cloudlog.exception("Upload error {url}")
    ###Upload GPS###


  def recompute_route(self):
    if self.last_position is None:
      return

    new_destination = coordinate_from_param("NavDestination", self.params)
    if new_destination is None:
      self.clear_route()
      self.reset_recompute_limits()
      return

    should_recompute = self.should_recompute()
    if new_destination != self.nav_destination:
      cloudlog.warning(f"Got new destination from NavDestination param {new_destination}")
      should_recompute = True

    # Don't recompute when GPS drifts in tunnels
    if not self.gps_ok and self.step_idx is not None:
      return

    if self.recompute_countdown == 0 and should_recompute:
      self.recompute_countdown = 2**self.recompute_backoff
      self.recompute_backoff = min(6, self.recompute_backoff + 1)
      self.calculate_route(new_destination)
      self.reroute_counter = 0
    else:
      self.recompute_countdown = max(0, self.recompute_countdown - 1)

  def calculate_route(self, destination):
    cloudlog.warning(f"Calculating route {self.last_position} -> {destination}")
    self.nav_destination = destination

    lang = self.params.get('LanguageSetting', encoding='utf8')
    if lang is not None:
      lang = lang.replace('main_', '')

    token = self.mapbox_token
    if token is None:
      token = self.api.get_token()

    params = {
      'access_token': token,
      'annotations': 'maxspeed',
      'geometries': 'geojson',
      'overview': 'full',
      'steps': 'true',
      'banner_instructions': 'true',
      'alternatives': 'false',
      'language': lang,
    }

    # TODO: move waypoints into NavDestination param?
    waypoints = self.params.get('NavDestinationWaypoints', encoding='utf8')
    waypoint_coords = []
    if waypoints is not None and len(waypoints) > 0:
      waypoint_coords = json.loads(waypoints)

    coords = [
      (self.last_position.longitude, self.last_position.latitude),
      *waypoint_coords,
      (destination.longitude, destination.latitude)
    ]
    params['waypoints'] = f'0;{len(coords)-1}'
    if self.last_bearing is not None:
      params['bearings'] = f"{(self.last_bearing + 360) % 360:.0f},90" + (';'*(len(coords)-1))

    coords_str = ';'.join([f'{lon},{lat}' for lon, lat in coords])
    url = self.mapbox_host + '/directions/v5/mapbox/driving-traffic/' + coords_str
    try:
      resp = requests.get(url, params=params, timeout=10)
      if resp.status_code != 200:
        cloudlog.event("API request failed", status_code=resp.status_code, text=resp.text, error=True)
      resp.raise_for_status()

      r = resp.json()
      if len(r['routes']):
        self.route = r['routes'][0]['legs'][0]['steps']
        self.route_geometry = []

        cloudlog.warning(f"route len steps {len(self.route)}")
        maxspeed_idx = 0
        maxspeeds = r['routes'][0]['legs'][0]['annotation']['maxspeed']

        # Convert coordinates
        for step in self.route:
          coords = []

          for c in step['geometry']['coordinates']:
            coord = Coordinate.from_mapbox_tuple(c)

            # Last step does not have maxspeed
            if (maxspeed_idx < len(maxspeeds)):
              maxspeed = maxspeeds[maxspeed_idx]
              if ('unknown' not in maxspeed) and ('none' not in maxspeed):
                coord.annotations['maxspeed'] = maxspeed_to_ms(maxspeed)

            coords.append(coord)
            maxspeed_idx += 1

          self.route_geometry.append(coords)
          maxspeed_idx -= 1  # Every segment ends with the same coordinate as the start of the next

        self.step_idx = 0
      else:
        cloudlog.warning("Got empty route response")
        self.clear_route()

      # clear waypoints to avoid a re-route including past waypoints
      # TODO: only clear once we're past a waypoint
      self.params.remove('NavDestinationWaypoints')

    except requests.exceptions.RequestException:
      cloudlog.exception("failed to get route")
      self.clear_route()

    self.send_route()

  def send_instruction(self):
    msg = messaging.new_message('navInstruction', valid=True)

    if self.step_idx is None:
      msg.valid = False
      self.pm.send('navInstruction', msg)
      return

    step = self.route[self.step_idx]
    geometry = self.route_geometry[self.step_idx]
    along_geometry = distance_along_geometry(geometry, self.last_position)
    distance_to_maneuver_along_geometry = step['distance'] - along_geometry

    # Banner instructions are for the following maneuver step, don't use empty last step
    banner_step = step
    if not len(banner_step['bannerInstructions']) and self.step_idx == len(self.route) - 1:
      banner_step = self.route[max(self.step_idx - 1, 0)]

    # Current instruction
    msg.navInstruction.maneuverDistance = distance_to_maneuver_along_geometry
    instruction = parse_banner_instructions(banner_step['bannerInstructions'], distance_to_maneuver_along_geometry)
    if instruction is not None:
      for k,v in instruction.items():
        setattr(msg.navInstruction, k, v)

    # All instructions
    maneuvers = []
    for i, step_i in enumerate(self.route):
      if i < self.step_idx:
        distance_to_maneuver = -sum(self.route[j]['distance'] for j in range(i+1, self.step_idx)) - along_geometry
      elif i == self.step_idx:
        distance_to_maneuver = distance_to_maneuver_along_geometry
      else:
        distance_to_maneuver = distance_to_maneuver_along_geometry + sum(self.route[j]['distance'] for j in range(self.step_idx+1, i+1))

      instruction = parse_banner_instructions(step_i['bannerInstructions'], distance_to_maneuver)
      if instruction is None:
        continue
      maneuver = {'distance': distance_to_maneuver}
      if 'maneuverType' in instruction:
        maneuver['type'] = instruction['maneuverType']
      if 'maneuverModifier' in instruction:
        maneuver['modifier'] = instruction['maneuverModifier']
      maneuvers.append(maneuver)

    msg.navInstruction.allManeuvers = maneuvers

    # Compute total remaining time and distance
    remaining = 1.0 - along_geometry / max(step['distance'], 1)
    total_distance = step['distance'] * remaining
    total_time = step['duration'] * remaining

    if step['duration_typical'] is None:
      total_time_typical = total_time
    else:
      total_time_typical = step['duration_typical'] * remaining

    # Add up totals for future steps
    for i in range(self.step_idx + 1, len(self.route)):
      total_distance += self.route[i]['distance']
      total_time += self.route[i]['duration']
      if self.route[i]['duration_typical'] is None:
        total_time_typical += self.route[i]['duration']
      else:
        total_time_typical += self.route[i]['duration_typical']

    msg.navInstruction.distanceRemaining = total_distance
    msg.navInstruction.timeRemaining = total_time
    msg.navInstruction.timeRemainingTypical = total_time_typical

    # Speed limit
    closest_idx, closest = min(enumerate(geometry), key=lambda p: p[1].distance_to(self.last_position))
    if closest_idx > 0:
      # If we are not past the closest point, show previous
      if along_geometry < distance_along_geometry(geometry, geometry[closest_idx]):
        closest = geometry[closest_idx - 1]

    if ('maxspeed' in closest.annotations) and self.localizer_valid:
      msg.navInstruction.speedLimit = closest.annotations['maxspeed']

    # Speed limit sign type
    if 'speedLimitSign' in step:
      if step['speedLimitSign'] == 'mutcd':
        msg.navInstruction.speedLimitSign = log.NavInstruction.SpeedLimitSign.mutcd
      elif step['speedLimitSign'] == 'vienna':
        msg.navInstruction.speedLimitSign = log.NavInstruction.SpeedLimitSign.vienna

    self.pm.send('navInstruction', msg)

    # Transition to next route segment
    if distance_to_maneuver_along_geometry < -MANEUVER_TRANSITION_THRESHOLD:
      if self.step_idx + 1 < len(self.route):
        self.step_idx += 1
        self.reset_recompute_limits()
      else:
        cloudlog.warning("Destination reached")

        # Clear route if driving away from destination
        dist = self.nav_destination.distance_to(self.last_position)
        if dist > REROUTE_DISTANCE:
          self.params.remove("NavDestination")
          self.clear_route()

  def send_route(self):
    coords = []

    if self.route is not None:
      for path in self.route_geometry:
        coords += [c.as_dict() for c in path]

    msg = messaging.new_message('navRoute', valid=True)
    msg.navRoute.coordinates = coords
    self.pm.send('navRoute', msg)

  def clear_route(self):
    self.route = None
    self.route_geometry = None
    self.step_idx = None
    self.nav_destination = None

  def reset_recompute_limits(self):
    self.recompute_backoff = 0
    self.recompute_countdown = 0

  def should_recompute(self):
    if self.step_idx is None or self.route is None:
      return True

    # Don't recompute in last segment, assume destination is reached
    if self.step_idx == len(self.route) - 1:
      return False

    x = []
    y = []
    path = self.route_geometry[self.step_idx]
    for i in range(len(path) - 1):
      x.append(path[i].longitude)
      y.append(path[i].latitude)

    y_val = self.last_position.latitude
    v = interp(y_val,y,x)
    v_point = Coordinate(y_val,v)
    distance = v_point.distance_to(self.last_position)

    if distance < REROUTE_DISTANCE*4:
      return False

    cloudlog.warning(f"should_recompute distanc={distance} v_point={v_point} last_position={self.last_position}")
    return True

    # Compute closest distance to all line segments in the current path
    min_d = REROUTE_DISTANCE + 1
    path = self.route_geometry[self.step_idx]
    for i in range(len(path) - 1):
      a = path[i]
      b = path[i + 1]

      if a.distance_to(b) < 1.0:
        continue

      min_d = min(min_d, minimum_distance(a, b, self.last_position))

    if min_d > REROUTE_DISTANCE:
      self.reroute_counter += 1
    else:
      self.reroute_counter = 0
    return self.reroute_counter > REROUTE_COUNTER_MIN
    # TODO: Check for going wrong way in segment


def main():
  pm = messaging.PubMaster(['navInstruction', 'navRoute'])
  sm = messaging.SubMaster(['liveLocationKalman', 'managerState'])

  rk = Ratekeeper(1.0)
  route_engine = RouteEngine(sm, pm)
  while True:
    route_engine.update()
    rk.keep_time()


if __name__ == "__main__":
  main()

