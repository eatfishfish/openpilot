import json
import mmap
import os
import time

import fcntl


class BlindSpotMonitor:
  def __init__(self):
    self.leftBlindspot = False
    self.rightBlindspot = False
    self.last_read = 0.0
    self.read_interval = 0.2
    self.state_timeout = 1.0
    self.leftBlindspot_timeout = 0.0
    self.rightBlindspot_timeout = 0.0
    self.left_shm_fd = None
    self.right_shm_fd = None
    self.left_shm_mm = None
    self.right_shm_mm = None
    self._init_shared_memory()

  def _init_shared_memory(self):
    self.left_shm_fd, self.left_shm_mm = self._open_shared_memory('/dev/shm/bsm_status_left')
    self.right_shm_fd, self.right_shm_mm = self._open_shared_memory('/dev/shm/bsm_status_right')

  def _open_shared_memory(self, path: str):
    try:
      shm_fd = os.open(path, os.O_RDWR)
      os.ftruncate(shm_fd, 1024)
      return shm_fd, mmap.mmap(shm_fd, 1024)
    except OSError as e:
      print(f"Error: unable to initialize {path}: {e}")
      return None, None

  def _read_shared_memory(self, side: str) -> dict | None:
    shm_fd = self.left_shm_fd if side == 'left' else self.right_shm_fd
    shm_mm = self.left_shm_mm if side == 'left' else self.right_shm_mm
    if shm_mm is None or shm_fd is None:
      return None

    try:
      fcntl.flock(shm_fd, fcntl.LOCK_SH)
      shm_mm.seek(0)
      data = shm_mm.read(1024).split(b'\0')[0].decode('utf-8')
      if not data:
        return None
      return json.loads(data)
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError, OSError) as e:
      print(f"Error: failed to read {side} blindspot data: {e}")
      return None
    finally:
      fcntl.flock(shm_fd, fcntl.LOCK_UN)

  def _update_side(self, side: str, current_time: float) -> bool:
    data = self._read_shared_memory(side)
    if not data or abs(current_time - float(data.get('timestamp', 0.0))) >= 0.1:
      return False

    blindspot = bool(data.get('blindspot', False))
    # if blindspot:
      # print(f"{side.capitalize()} blindspot={blindspot}, distance={data.get('distance')}m")
    return blindspot

  def update(self) -> tuple[bool, bool]:
    current_time = time.time()
    if current_time - self.last_read >= self.read_interval:
      self.leftBlindspot = self._update_side('left', current_time)
      self.rightBlindspot = self._update_side('right', current_time)
      self.leftBlindspot_timeout = current_time + self.state_timeout if self.leftBlindspot else 0.0
      self.rightBlindspot_timeout = current_time + self.state_timeout if self.rightBlindspot else 0.0
      self.last_read = current_time

    if current_time > self.leftBlindspot_timeout:
      self.leftBlindspot = False
    if current_time > self.rightBlindspot_timeout:
      self.rightBlindspot = False

    return self.leftBlindspot, self.rightBlindspot

  def __del__(self):
    if self.left_shm_mm is not None:
      self.left_shm_mm.close()
    if self.right_shm_mm is not None:
      self.right_shm_mm.close()
    if self.left_shm_fd is not None:
      os.close(self.left_shm_fd)
    if self.right_shm_fd is not None:
      os.close(self.right_shm_fd)
