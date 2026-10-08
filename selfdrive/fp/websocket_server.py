#!/usr/bin/env python3
import asyncio
import base64
from collections import deque
import hashlib
import os
import struct
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from openpilot.common.params import Params

HOST = os.getenv("H264_WS_HOST", "0.0.0.0")
PORT = int(os.getenv("H264_WS_PORT", "8089"))
PUSH_FPS = float(os.getenv("H264_WS_FPS", "20"))
ATHENA_HOST = os.getenv("ATHENA_HOST", "")
REMOTE_MAX_LATENCY = float(os.getenv("H264_REMOTE_MAX_LATENCY", "1.5"))
HTML_PATH = Path(__file__).with_name("viewer.html")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def get_dongle_id():
  return os.getenv("DONGLE_ID") or Params().get("DongleId") or ""


class MjpegWebsocketServer:
  def __init__(self, push_fps):
    self.push_fps = push_fps
    self.viewers = {}
    self.ingests = {}
    self.remote_viewers = {}
    self.remote_clients = {}
    self.dongle_id = get_dongle_id()

  async def handle_client(self, reader, writer):
    try:
      request = await self.read_http_request(reader)
      if not request:
        return
      method, path, headers = request
      if headers.get("upgrade", "").lower() == "websocket":
        await self.handle_websocket(path, headers, reader, writer)
      else:
        await self.handle_http(method, path, writer)
    finally:
      writer.close()
      await writer.wait_closed()

  async def read_http_request(self, reader):
    data = await reader.readuntil(b"\r\n\r\n")
    lines = data.decode("iso-8859-1").split("\r\n")
    parts = lines[0].split()
    if len(parts) < 2:
      return None
    headers = {}
    for line in lines[1:]:
      if ":" in line:
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return parts[0], parts[1], headers

  async def handle_http(self, method, path, writer):
    route = urlsplit(path).path
    if method != "GET" or route not in ("/", "/viewer.html"):
      await self.send_http(writer, "404 Not Found", b"not found", "text/plain")
      return
    body = HTML_PATH.read_bytes()
    await self.send_http(writer, "200 OK", body, "text/html; charset=utf-8")

  async def send_http(self, writer, status, body, content_type):
    writer.write(
      f"HTTP/1.1 {status}\r\n"
      f"Content-Type: {content_type}\r\n"
      f"Content-Length: {len(body)}\r\n"
      "Connection: close\r\n\r\n".encode("ascii") + body
    )
    await writer.drain()

  async def handle_websocket(self, path, headers, reader, writer):
    key = headers.get("sec-websocket-key")
    url = urlsplit(path)
    route = url.path
    camera = parse_qs(url.query).get("camera", ["roadCameraState"])[0]
    if not key or route not in ("/stream", "/ingest"):
      writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
      await writer.drain()
      return

    accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")
    writer.write(
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode("ascii")
    )
    await writer.drain()

    if route == "/stream":
      await self.viewer_loop(camera, reader, writer)
    else:
      await self.ingest_loop(camera, reader, writer)

  async def viewer_loop(self, camera, reader, writer):
    self.viewers.setdefault(camera, set()).add(writer)
    await self.notify_ingests(camera)
    await self.request_keyframe(camera)
    try:
      while True:
        opcode, _ = await self.read_ws_frame(reader)
        if opcode == 0x8:
          break
    finally:
      self.viewers.get(camera, set()).discard(writer)
      await self.notify_ingests(camera)

  async def ingest_loop(self, camera, reader, writer):
    self.ingests.setdefault(camera, set()).add(writer)
    await self.notify_ingests(camera)
    try:
      while True:
        opcode, payload = await self.read_ws_frame(reader)
        if opcode == 0x8:
          break
        if opcode == 0x2:
          cmd, body = self.parse_ingest_packet(payload)
          if cmd == 0:
            old_camera = camera
            camera = body.decode("utf-8", errors="ignore") or camera
            if old_camera != camera:
              self.ingests.get(old_camera, set()).discard(writer)
              await self.notify_ingests(old_camera)
            self.ingests.setdefault(camera, set()).add(writer)
            await self.notify_ingests(camera)
          elif cmd == 1:
            await self.broadcast_frame(camera, body)
    finally:
      self.ingests.get(camera, set()).discard(writer)

  async def notify_ingests(self, camera):
    has_viewer = bool(self.viewers.get(camera)) or self.remote_viewers.get(camera, False)
    message = self.make_ingest_packet(2, b"1" if has_viewer else b"0")
    stale = []
    for writer in self.ingests.get(camera, set()):
      try:
        await self.write_ws_frame(writer, message, opcode=0x2)
      except (ConnectionError, OSError):
        stale.append(writer)
    for writer in stale:
      self.ingests.get(camera, set()).discard(writer)

  async def request_keyframe(self, camera):
    message = self.make_ingest_packet(3, b"")
    stale = []
    for writer in self.ingests.get(camera, set()):
      try:
        await self.write_ws_frame(writer, message, opcode=0x2)
      except (ConnectionError, OSError):
        stale.append(writer)
    for writer in stale:
      self.ingests.get(camera, set()).discard(writer)

  def parse_ingest_packet(self, payload):
    if len(payload) < 4:
      return None, b""
    cmd = payload[0]
    length = (payload[1] << 16) | (payload[2] << 8) | payload[3]
    if 4 + length > len(payload):
      return None, b""
    return cmd, payload[4:4 + length]

  def make_ingest_packet(self, cmd, payload):
    if len(payload) > 0xffffff:
      raise ValueError("payload too large")
    return bytes((cmd, (len(payload) >> 16) & 0xff, (len(payload) >> 8) & 0xff, len(payload) & 0xff)) + payload

  @staticmethod
  def is_h264_keyframe(payload):
    index = 0
    while index + 4 <= len(payload):
      if payload[index:index + 4] == b"\x00\x00\x00\x01":
        nal_start = index + 4
        index = nal_start
      elif payload[index:index + 3] == b"\x00\x00\x01":
        nal_start = index + 3
        index = nal_start
      else:
        index += 1
        continue
      if nal_start < len(payload) and payload[nal_start] & 0x1f == 5:
        return True
    return False

  async def broadcast_frame(self, camera, payload):
    viewers = self.viewers.get(camera, set())
    remote_client = await self.ensure_remote_client(camera)
    if not viewers and not self.remote_viewers.get(camera, False):
      return

    stale = []
    for writer in viewers:
      try:
        await self.write_ws_frame(writer, payload, opcode=0x2)
      except (ConnectionError, OSError):
        stale.append(writer)
    for writer in stale:
      viewers.discard(writer)
    if stale:
      await self.notify_ingests(camera)
    if remote_client and self.remote_viewers.get(camera, False):
      await remote_client.send_h264(payload)

  async def read_ws_frame(self, reader):
    header = await reader.readexactly(2)
    opcode = header[0] & 0x0f
    masked = (header[1] & 0x80) != 0
    length = header[1] & 0x7f
    if length == 126:
      length = struct.unpack("!H", await reader.readexactly(2))[0]
    elif length == 127:
      length = struct.unpack("!Q", await reader.readexactly(8))[0]

    mask = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(length) if length else b""
    if masked:
      payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload

  async def write_ws_frame(self, writer, payload, opcode=0x2):
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length <= 125:
      header.append(length)
    elif length <= 0xffff:
      header.extend((126, *struct.pack("!H", length)))
    else:
      header.extend((127, *struct.pack("!Q", length)))
    writer.write(bytes(header) + payload)
    await writer.drain()

  async def ensure_remote_client(self, camera):
    if not ATHENA_HOST:
      return None
    client = self.remote_clients.get(camera)
    if client:
      return client
    client = RemoteH264Client(ATHENA_HOST, self.dongle_id, camera, self)
    self.remote_clients[camera] = client
    asyncio.create_task(client.run())
    return client

  async def start_remote_clients(self):
    if not ATHENA_HOST:
      return
    for camera in ("roadCameraState", "wideRoadCameraState"):
      await self.ensure_remote_client(camera)


class RemoteH264Client:
  def __init__(self, athena_host, dongle_id, camera, server):
    self.athena_host = athena_host
    self.dongle_id = dongle_id
    self.camera = camera
    self.server = server
    self.reader = None
    self.writer = None
    self.connected = False
    self.write_lock = asyncio.Lock()
    self.pending_frames = deque()
    self.pending_event = asyncio.Event()
    self.waiting_for_keyframe = True

  async def run(self):
    while True:
      try:
        await self.connect()
        sender_task = asyncio.create_task(self.send_loop())
        reader_task = asyncio.create_task(self.read_loop())
        try:
          done, pending = await asyncio.wait(
            (reader_task, sender_task),
            return_when=asyncio.FIRST_COMPLETED,
          )
          for task in pending:
            task.cancel()
          await asyncio.gather(*pending, return_exceptions=True)
          for task in done:
            task.result()
        finally:
          sender_task.cancel()
          reader_task.cancel()
      except (ConnectionError, OSError, asyncio.IncompleteReadError, asyncio.TimeoutError):
        pass
      finally:
        self.connected = False
        self.waiting_for_keyframe = True
        self.clear_pending_frames()
        self.server.remote_viewers[self.camera] = False
        await self.server.notify_ingests(self.camera)
        if self.writer:
          self.writer.close()
          await self.writer.wait_closed()
      await asyncio.sleep(2)

  async def connect(self):
    url = self.normalize_url()
    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    ssl_enabled = parsed.scheme == "wss"
    self.reader, self.writer = await asyncio.open_connection(parsed.hostname, port, ssl=ssl_enabled)
    base_path = (parsed.path or "").rstrip("/")
    path = f"{base_path}/h264/ingest?dongle_id={self.dongle_id}&camera={self.camera}"
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
      f"GET {path} HTTP/1.1\r\n"
      f"Host: {parsed.netloc}\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      f"Sec-WebSocket-Key: {key}\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "\r\n"
    )
    self.writer.write(request.encode("ascii"))
    await self.writer.drain()
    response = await self.reader.readuntil(b"\r\n\r\n")
    if b" 101 " not in response:
      raise ConnectionError("remote websocket upgrade failed")
    self.connected = True
    await self.send_packet(0, self.camera.encode("utf-8"))
    if self.server.remote_viewers.get(self.camera, False):
      await self.server.request_keyframe(self.camera)

  def normalize_url(self):
    if self.athena_host.startswith(("ws://", "wss://", "http://", "https://")):
      return self.athena_host.replace("http://", "ws://", 1).replace("https://", "wss://", 1).rstrip("/")
    return "ws://" + self.athena_host.rstrip("/")

  async def read_loop(self):
    while True:
      opcode, payload = await self.server.read_ws_frame(self.reader)
      if opcode == 0x8:
        raise ConnectionError("remote closed")
      if opcode == 0x2:
        cmd, body = self.server.parse_ingest_packet(payload)
        if cmd == 2 and body:
          had_viewer = self.server.remote_viewers.get(self.camera, False)
          self.server.remote_viewers[self.camera] = body[:1] == b"1"
          await self.server.notify_ingests(self.camera)
          if not had_viewer and self.server.remote_viewers[self.camera]:
            self.waiting_for_keyframe = True
            self.clear_pending_frames()
            await self.server.request_keyframe(self.camera)
        elif cmd == 3:
          self.waiting_for_keyframe = True
          self.clear_pending_frames()
          await self.server.request_keyframe(self.camera)

  async def send_h264(self, payload):
    is_keyframe = self.server.is_h264_keyframe(payload)
    if self.waiting_for_keyframe:
      if not is_keyframe:
        return
      self.waiting_for_keyframe = False

    now = time.monotonic()
    if self.pending_frames and now - self.pending_frames[0][0] > REMOTE_MAX_LATENCY:
      self.clear_pending_frames()
      if not is_keyframe:
        self.waiting_for_keyframe = True
        await self.server.request_keyframe(self.camera)
        return

    self.pending_frames.append((now, payload))
    self.pending_event.set()

  def clear_pending_frames(self):
    self.pending_frames.clear()
    self.pending_event.clear()

  async def send_loop(self):
    while True:
      await self.pending_event.wait()
      while self.pending_frames:
        queued_at, frame = self.pending_frames.popleft()
        if time.monotonic() - queued_at > REMOTE_MAX_LATENCY:
          self.waiting_for_keyframe = True
          self.clear_pending_frames()
          await self.server.request_keyframe(self.camera)
          break
        await asyncio.wait_for(
          self.send_packet(1, frame),
          timeout=REMOTE_MAX_LATENCY,
        )
      if not self.pending_frames:
        self.pending_event.clear()

  async def send_packet(self, cmd, payload):
    if not self.connected or not self.writer:
      return
    packet = self.server.make_ingest_packet(cmd, payload)
    async with self.write_lock:
      await self.write_client_ws_frame(packet, opcode=0x2)

  async def write_client_ws_frame(self, payload, opcode=0x2):
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length <= 125:
      header.append(0x80 | length)
    elif length <= 0xffff:
      header.extend((0x80 | 126, *struct.pack("!H", length)))
    else:
      header.extend((0x80 | 127, *struct.pack("!Q", length)))

    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    self.writer.write(bytes(header) + mask + masked)
    await self.writer.drain()


async def async_main():
  server = MjpegWebsocketServer(PUSH_FPS)
  await server.start_remote_clients()
  tcp_server = await asyncio.start_server(server.handle_client, HOST, PORT)
  print(f"H.264 websocket server listening on http://{HOST}:{PORT}, fps={PUSH_FPS:g}")
  async with tcp_server:
    await tcp_server.serve_forever()


def main():
  asyncio.run(async_main())


if __name__ == "__main__":
  main()
