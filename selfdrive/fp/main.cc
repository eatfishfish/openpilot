#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "common/util.h"
#include "msgq/visionipc/visionipc_client.h"
#ifdef QCOM2
#include "selfdrive/fp/streamup845.h"
#else
#include "selfdrive/fp/streamup.h"
#endif

namespace {

constexpr int DEFAULT_FPS = 20;
constexpr int DEFAULT_BITRATE = 2'000'000;

struct StreamConfig {
  VisionStreamType type;
  const char *camera;
  const char *thread_name;
};

void stream_camera(const StreamConfig config, ExitHandler *do_exit) {
  util::set_thread_name(config.thread_name);

  while (!*do_exit) {
    VisionIpcClient client("camerad", config.type, true);
    if (!client.connect(false)) {
      util::sleep_for(200);
      continue;
    }

    const VisionBuf &info = client.buffers[0];
    if (info.width == 0 || info.height == 0) {
      fprintf(stderr, "%s invalid VisionBuf size\n", config.camera);
      util::sleep_for(1000);
      continue;
    }

    fprintf(stderr, "%s H.264 streamer connected: %zux%zu\n",
            config.camera, info.width, info.height);

    AmdH264VaapiEncoder encoder(
        static_cast<int>(info.width),
        static_cast<int>(info.height),
        config.camera,
        DEFAULT_FPS,
        DEFAULT_BITRATE,
        "");

    while (!*do_exit && client.is_connected()) {
      VisionIpcBufExtra extra = {};
      VisionBuf *buf = client.recv(&extra, 1000);
      if (!buf) {
        continue;
      }
      if (buf->get_frame_id() != extra.frame_id) {
        continue;
      }

      if (!encoder.encode_frame(buf)) {
        util::sleep_for(50);
      }
    }

    fprintf(stderr, "%s VisionIPC disconnected, reconnecting\n", config.camera);
  }
}

}  // namespace

int main() {
  ExitHandler do_exit;
  const std::vector<StreamConfig> streams = {
      {VISION_STREAM_ROAD, "roadCameraState", "h264_road"},
      {VISION_STREAM_WIDE_ROAD, "wideRoadCameraState", "h264_wide"},
  };

  std::vector<std::thread> threads;
  threads.reserve(streams.size());
  for (const StreamConfig &config : streams) {
    threads.emplace_back(stream_camera, config, &do_exit);
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  return 0;
}
