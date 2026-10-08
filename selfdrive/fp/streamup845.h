#pragma once

// Snapdragon 845 H.264 encoding through FFmpeg.
// Input NV12 frames are converted to I420 with libyuv before encoding.

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "msgq/visionipc/visionbuf.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
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
  std::string camera_;
  std::string host_;
  int port_ = 8089;

  AVCodecContext *codec_ctx_ = nullptr;
  AVFrame *sw_frame_ = nullptr;
  const AVCodec *codec_ = nullptr;
  std::vector<uint8_t> convert_buf_;

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
