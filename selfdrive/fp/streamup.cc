#include "selfdrive/fp/streamup.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/swaglog.h"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

namespace {

constexpr auto RECONNECT_DELAY = std::chrono::seconds(2);

static std::string default_vaapi_device() {
  if (const char *env = getenv("VAAPI_DEVICE"); env && env[0] != '\0') {
    return env;
  }
  return "/dev/dri/renderD128";
}

bool write_all(int fd, const uint8_t *data, size_t size) {
  while (size > 0) {
    ssize_t written = send(fd, data, size, MSG_NOSIGNAL);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    data += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

}  // namespace

AmdH264VaapiEncoder::AmdH264VaapiEncoder(int width, int height, const std::string &camera,
                                         int fps, int bitrate, const std::string &device_path,
                                         const std::string &host, int port)
    : width_(width),
      height_(height),
      fps_(fps),
      bitrate_(bitrate),
      device_path_(device_path.empty() ? default_vaapi_device() : device_path),
      camera_(camera),
      host_(host),
      port_(port),
      next_reconnect_(std::chrono::steady_clock::now()) {
  if (const char *host_env = getenv("H264_WS_HOST"); host_env && host_env[0] != '\0') {
    host_ = host_env;
  } else if (const char *legacy_host_env = getenv("H264_WS_HOST"); legacy_host_env && legacy_host_env[0] != '\0') {
    host_ = legacy_host_env;
  }
  if (const char *port_env = getenv("H264_WS_PORT"); port_env && port_env[0] != '\0') {
    port_ = std::max(1, atoi(port_env));
  } else if (const char *legacy_port_env = getenv("H264_WS_PORT"); legacy_port_env && legacy_port_env[0] != '\0') {
    port_ = std::max(1, atoi(legacy_port_env));
  }
  if (const char *fps_env = getenv("H264_WS_FPS"); fps_env && fps_env[0] != '\0') {
    fps_ = std::max(1, atoi(fps_env));
  } else if (const char *legacy_fps_env = getenv("H264_WS_FPS"); legacy_fps_env && legacy_fps_env[0] != '\0') {
    fps_ = std::max(1, atoi(legacy_fps_env));
  }
  if (const char *bitrate_env = getenv("H264_WS_BITRATE"); bitrate_env && bitrate_env[0] != '\0') {
    bitrate_ = std::max(1, atoi(bitrate_env));
  }
  gop_size_ = fps_ > 0 ? fps_ * 2 : 40;
  hw_frame_ = av_frame_alloc();
  sw_frame_ = av_frame_alloc();
}

AmdH264VaapiEncoder::~AmdH264VaapiEncoder() {
  close_socket();
  close();
  av_frame_free(&hw_frame_);
  av_frame_free(&sw_frame_);
}

bool AmdH264VaapiEncoder::init_hw_device() {
  if (hw_device_ctx_) {
    av_buffer_unref(&hw_device_ctx_);
  }
  int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_VAAPI, device_path_.c_str(), nullptr, 0);
  if (ret < 0) {
    LOGE("VAAPI device open failed: %s (%d)", device_path_.c_str(), ret);
    return false;
  }
  return true;
}

bool AmdH264VaapiEncoder::init_hw_frames() {
  if (hw_frames_ctx_) {
    av_buffer_unref(&hw_frames_ctx_);
  }

  AVBufferRef *frames_ref = av_hwframe_ctx_alloc(hw_device_ctx_);
  if (!frames_ref) {
    LOGE("av_hwframe_ctx_alloc failed");
    return false;
  }

  auto *frames_ctx = reinterpret_cast<AVHWFramesContext *>(frames_ref->data);
  frames_ctx->format = AV_PIX_FMT_VAAPI;
  frames_ctx->sw_format = AV_PIX_FMT_NV12;
  frames_ctx->width = width_;
  frames_ctx->height = height_;
  frames_ctx->initial_pool_size = 8;

  int ret = av_hwframe_ctx_init(frames_ref);
  if (ret < 0) {
    LOGE("av_hwframe_ctx_init failed: %d", ret);
    av_buffer_unref(&frames_ref);
    return false;
  }

  hw_frames_ctx_ = frames_ref;
  return true;
}

bool AmdH264VaapiEncoder::open() {
  if (opened_) {
    return true;
  }

  codec_ = avcodec_find_encoder_by_name("h264_vaapi");
  if (!codec_) {
    LOGE("h264_vaapi encoder not found");
    return false;
  }

  if (!init_hw_device()) {
    release();
    return false;
  }
  if (!init_hw_frames()) {
    release();
    return false;
  }

  codec_ctx_ = avcodec_alloc_context3(codec_);
  if (!codec_ctx_) {
    LOGE("avcodec_alloc_context3 failed");
    release();
    return false;
  }

  codec_ctx_->width = width_;
  codec_ctx_->height = height_;
  codec_ctx_->time_base = AVRational{1, fps_};
  codec_ctx_->framerate = AVRational{fps_, 1};
  codec_ctx_->pix_fmt = AV_PIX_FMT_VAAPI;
  codec_ctx_->bit_rate = bitrate_;
  codec_ctx_->gop_size = gop_size_;
  codec_ctx_->max_b_frames = 0;
  codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
  codec_ctx_->color_range = AVCOL_RANGE_MPEG;
  codec_ctx_->colorspace = AVCOL_SPC_BT709;
  codec_ctx_->color_primaries = AVCOL_PRI_BT709;
  codec_ctx_->color_trc = AVCOL_TRC_BT709;

  if (!codec_ctx_->hw_device_ctx || !codec_ctx_->hw_frames_ctx) {
    LOGE("failed to ref hw contexts");
    close();
    return false;
  }

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "rc_mode", "VBR", 0);
  av_dict_set(&opts, "quality", "7", 0);

  int ret = avcodec_open2(codec_ctx_, codec_, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    LOGE("avcodec_open2(h264_vaapi) failed: %d", ret);
    release();
    return false;
  }

  opened_ = true;
  frame_idx_ = 0;
  return true;
}

void AmdH264VaapiEncoder::release() {
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
  }
  if (hw_frames_ctx_) {
    av_buffer_unref(&hw_frames_ctx_);
  }
  if (hw_device_ctx_) {
    av_buffer_unref(&hw_device_ctx_);
  }
  opened_ = false;
}

void AmdH264VaapiEncoder::close() {
  if (!opened_ && !codec_ctx_) {
    return;
  }

  if (codec_ctx_) {
    avcodec_send_frame(codec_ctx_, nullptr);
    drain_packets(nullptr);
  }

  release();
}

bool AmdH264VaapiEncoder::drain_packets(std::vector<uint8_t> *out) {
  AVPacket *pkt = av_packet_alloc();
  if (!pkt) {
    LOGE("av_packet_alloc failed");
    return false;
  }

  while (true) {
    int ret = avcodec_receive_packet(codec_ctx_, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_packet_free(&pkt);
      return true;
    }
    if (ret < 0) {
      LOGE("avcodec_receive_packet failed: %d", ret);
      av_packet_free(&pkt);
      return false;
    }
    if (out) {
      out->insert(out->end(), pkt->data, pkt->data + pkt->size);
    }
    av_packet_unref(pkt);
  }
}

bool AmdH264VaapiEncoder::ensure_connected() {
  if (socket_fd_ >= 0) {
    return true;
  }

  const auto now = std::chrono::steady_clock::now();
  if (now < next_reconnect_) {
    return false;
  }
  next_reconnect_ = now + RECONNECT_DELAY;

  addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  addrinfo *result = nullptr;
  const std::string port = std::to_string(port_);
  if (getaddrinfo(host_.c_str(), port.c_str(), &hints, &result) != 0) {
    return false;
  }

  for (addrinfo *rp = result; rp != nullptr; rp = rp->ai_next) {
    int fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) {
      continue;
    }
    if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
      socket_fd_ = fd;
      break;
    }
    ::close(fd);
  }
  freeaddrinfo(result);

  if (socket_fd_ < 0) {
    return false;
  }
  if (!send_handshake() || !send_camera_register()) {
    close_socket();
    return false;
  }

  has_viewer_ = false;
  restart_encoder_ = true;
  return true;
}

bool AmdH264VaapiEncoder::send_handshake() {
  const std::string path = "/ingest?camera=" + camera_;
  const std::string request =
      "GET " + path + " HTTP/1.1\r\n"
      "Host: " + host_ + ":" + std::to_string(port_) + "\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Key: ZHJhZ29ucGlsb3QtaDI2NA==\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "\r\n";
  if (!write_all(socket_fd_, reinterpret_cast<const uint8_t *>(request.data()), request.size())) {
    return false;
  }

  std::string response;
  char buffer[512];
  while (response.find("\r\n\r\n") == std::string::npos && response.size() < 4096) {
    const ssize_t size = recv(socket_fd_, buffer, sizeof(buffer), 0);
    if (size <= 0) {
      return false;
    }
    response.append(buffer, static_cast<size_t>(size));
  }
  return response.find(" 101 ") != std::string::npos;
}

bool AmdH264VaapiEncoder::send_camera_register() {
  return send_packet(0, reinterpret_cast<const uint8_t *>(camera_.data()), camera_.size());
}

bool AmdH264VaapiEncoder::send_packet(uint8_t cmd, const uint8_t *data, size_t size) {
  if (size > 0xffffff) {
    LOGE("H.264 websocket payload too large: %zu", size);
    return false;
  }

  std::vector<uint8_t> packet;
  packet.reserve(size + 4);
  packet.push_back(cmd);
  packet.push_back(static_cast<uint8_t>((size >> 16) & 0xff));
  packet.push_back(static_cast<uint8_t>((size >> 8) & 0xff));
  packet.push_back(static_cast<uint8_t>(size & 0xff));
  packet.insert(packet.end(), data, data + size);
  return send_ws_binary(packet.data(), packet.size());
}

bool AmdH264VaapiEncoder::send_ws_binary(const uint8_t *data, size_t size) {
  std::vector<uint8_t> header = {0x82};
  if (size <= 125) {
    header.push_back(static_cast<uint8_t>(size));
  } else if (size <= 0xffff) {
    header.push_back(126);
    header.push_back(static_cast<uint8_t>((size >> 8) & 0xff));
    header.push_back(static_cast<uint8_t>(size & 0xff));
  } else {
    header.push_back(127);
    for (int shift = 56; shift >= 0; shift -= 8) {
      header.push_back(static_cast<uint8_t>((static_cast<uint64_t>(size) >> shift) & 0xff));
    }
  }
  return write_all(socket_fd_, header.data(), header.size()) && write_all(socket_fd_, data, size);
}

void AmdH264VaapiEncoder::read_control_messages() {
  if (socket_fd_ < 0) {
    return;
  }

  while (true) {
    uint8_t data[4096];
    const ssize_t size = recv(socket_fd_, data, sizeof(data), MSG_DONTWAIT);
    if (size == 0) {
      close_socket();
      return;
    }
    if (size < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        break;
      }
      close_socket();
      return;
    }
    rx_buffer_.insert(rx_buffer_.end(), data, data + size);
  }

  size_t offset = 0;
  while (rx_buffer_.size() - offset >= 2) {
    const uint8_t *header = rx_buffer_.data() + offset;
    const uint8_t opcode = header[0] & 0x0f;
    const bool masked = (header[1] & 0x80) != 0;
    uint64_t payload_size = header[1] & 0x7f;
    size_t header_size = 2;

    if (payload_size == 126) {
      if (rx_buffer_.size() - offset < 4) break;
      payload_size = (static_cast<uint64_t>(header[2]) << 8) | header[3];
      header_size = 4;
    } else if (payload_size == 127) {
      if (rx_buffer_.size() - offset < 10) break;
      payload_size = 0;
      for (size_t index = 2; index < 10; ++index) {
        payload_size = (payload_size << 8) | header[index];
      }
      header_size = 10;
    }

    if (payload_size > 1024 * 1024) {
      close_socket();
      return;
    }

    const size_t mask_size = masked ? 4 : 0;
    const uint64_t frame_size = header_size + mask_size + payload_size;
    if (frame_size > rx_buffer_.size() - offset) break;

    const uint8_t *mask = masked ? header + header_size : nullptr;
    const uint8_t *payload_data = header + header_size + mask_size;
    std::vector<uint8_t> payload(payload_data, payload_data + payload_size);
    if (masked) {
      for (size_t index = 0; index < payload.size(); ++index) {
        payload[index] ^= mask[index % 4];
      }
    }
    offset += static_cast<size_t>(frame_size);

    if (opcode == 0x8) {
      close_socket();
      return;
    }
    if ((opcode == 0x1 || opcode == 0x2) && payload.size() >= 5 && payload[0] == 2) {
      const bool has_viewer = payload[4] == '1';
      if (has_viewer && !has_viewer_) {
        restart_encoder_ = true;
      }
      has_viewer_ = has_viewer;
    } else if ((opcode == 0x1 || opcode == 0x2) && payload.size() >= 4 && payload[0] == 3) {
      restart_encoder_ = true;
    }
  }

  if (offset > 0) {
    rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + offset);
  }
}

void AmdH264VaapiEncoder::close_socket() {
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
  has_viewer_ = false;
  restart_encoder_ = true;
  rx_buffer_.clear();
}

bool AmdH264VaapiEncoder::fps_limited() const {
  if (fps_ <= 0 || last_sent_.time_since_epoch().count() == 0) {
    return false;
  }
  const auto interval = std::chrono::microseconds(1000000 / fps_);
  return std::chrono::steady_clock::now() - last_sent_ < interval;
}

bool AmdH264VaapiEncoder::encode_frame(const VisionBuf *buf) {
  if (!buf) {
    return false;
  }

  if (!ensure_connected()) {
    return true;
  }
  read_control_messages();
  if (!has_viewer_ || fps_limited()) {
    return true;
  }

  bool force_keyframe = false;
  if (restart_encoder_) {
    close();
    force_keyframe = true;
    restart_encoder_ = false;
  }

  if (!opened_ && !open()) {
    restart_encoder_ = true;
    return false;
  }
  if (buf->width != static_cast<size_t>(width_) || buf->height != static_cast<size_t>(height_)) {
    LOGE("input size mismatch: got %zux%zu expect %dx%d", buf->width, buf->height, width_, height_);
    return false;
  }

  av_frame_unref(sw_frame_);
  av_frame_unref(hw_frame_);

  sw_frame_->format = AV_PIX_FMT_NV12;
  sw_frame_->width = width_;
  sw_frame_->height = height_;
  sw_frame_->data[0] = buf->y;
  sw_frame_->data[1] = buf->uv;
  sw_frame_->linesize[0] = buf->stride;
  sw_frame_->linesize[1] = buf->stride;
  sw_frame_->pts = frame_idx_;

  if (force_keyframe || (gop_size_ > 0 && (frame_idx_ % gop_size_ == 0))) {
    sw_frame_->pict_type = AV_PICTURE_TYPE_I;
  }

  if (!hw_frame_) {
    hw_frame_ = av_frame_alloc();
    if (!hw_frame_) {
      return false;
    }
  }

  int ret = av_hwframe_get_buffer(hw_frames_ctx_, hw_frame_, 0);
  if (ret < 0) {
    LOGE("av_hwframe_get_buffer failed: %d", ret);
    return false;
  }
  ret = av_hwframe_transfer_data(hw_frame_, sw_frame_, 0);
  if (ret < 0) {
    LOGE("av_hwframe_transfer_data failed: %d", ret);
    return false;
  }

  hw_frame_->pts = sw_frame_->pts;
  hw_frame_->pict_type = sw_frame_->pict_type;

  ret = avcodec_send_frame(codec_ctx_, hw_frame_);
  if (ret < 0) {
    LOGE("avcodec_send_frame failed: %d", ret);
    return false;
  }

  std::vector<uint8_t> encoded;
  if (!drain_packets(&encoded)) {
    return false;
  }

  ++frame_idx_;
  if (encoded.empty()) {
    return true;
  }
  if (!send_packet(1, encoded.data(), encoded.size())) {
    close_socket();
    return false;
  }
  last_sent_ = std::chrono::steady_clock::now();
  return true;
}
