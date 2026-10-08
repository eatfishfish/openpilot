#pragma once

// Ubuntu 24.04 AMD VAAPI H.264 dependencies.
// This module links through FFmpeg; libva packages provide runtime VAAPI support/tools.
// sudo apt install -y ffmpeg libavcodec-dev libavutil-dev libva-dev libva-drm2 vainfo mesa-va-drivers
// Optional check:
// vainfo --display drm --device /dev/dri/renderD128

#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

#include "msgq/visionipc/visionbuf.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

class AmdH264VaapiEncoder {
public:
  AmdH264VaapiEncoder(int width, int height, const std::string &camera,
                      int fps = 20,
                      int bitrate = 5'000'000,
                      const std::string &device_path = "",
                      const std::string &host = "127.0.0.1",
                      int port = 8089);
  ~AmdH264VaapiEncoder();

  bool open();
  void close();
  bool encode_frame(const VisionBuf *buf);

  bool is_open() const { return opened_; }

private:
  bool init_hw_device();
  bool init_hw_frames();
  bool drain_packets(std::vector<uint8_t> *out);
  bool ensure_connected();
  bool send_handshake();
  bool send_camera_register();
  bool send_packet(uint8_t cmd, const uint8_t *data, size_t size);
  bool send_ws_binary(const uint8_t *data, size_t size);
  void read_control_messages();
  void close_socket();
  bool fps_limited() const;
  void release();

  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;
  int bitrate_ = 0;
  std::string device_path_;
  std::string camera_;
  std::string host_;
  int port_ = 8089;

  AVBufferRef *hw_device_ctx_ = nullptr;
  AVBufferRef *hw_frames_ctx_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVFrame *hw_frame_ = nullptr;
  AVFrame *sw_frame_ = nullptr;
  const AVCodec *codec_ = nullptr;

  bool opened_ = false;
  int socket_fd_ = -1;
  bool has_viewer_ = false;
  bool restart_encoder_ = true;
  int64_t frame_idx_ = 0;
  int gop_size_ = 0;
  std::vector<uint8_t> rx_buffer_;
  std::chrono::steady_clock::time_point last_sent_;
  std::chrono::steady_clock::time_point next_reconnect_;
};
