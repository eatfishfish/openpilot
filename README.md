# openpilot（ubuntu-AMD 分支）

这是一个个人维护的 openpilot 派生版本，**不是** comma.ai 的官方发行版，也不是给 comma 3/3X 用的安装包。

本分支（`ubuntu-AMD`）面向 **Ubuntu 平台的 AMD 硬件**，而不是 AGNOS / comma 3/3X 设备。

## 与上游的关系

```
commaai/openpilot  ──派生──▶  dragonpilot  ──派生──▶  本仓库
```

- 基座提交：`1508b2a6e31204c2fb41dd023412c90808236940`（2025-08-19，记录在 `git_src_commit` 中）
- 代码结构与 dragonpilot 一致，本仓库在其基础上继续修改
- 与 dragonpilot 的差异见下方「本仓库的改动」和提交历史

## 本仓库的改动

| 内容 | 说明 |
|---|---|
| YOLO 前车检测 | `selfdrive/modeld/yolo.py`、`modeld.py`、`tests/test_yolo.py` 对齐 dragonpilot 的版本，包含 `YoloLaneFilter` 车道线过滤 |
| vision-only 兜底设计 | 见 [`YOLO_VISION_ONLY_FALLBACK.md`](YOLO_VISION_ONLY_FALLBACK.md)，为 YOLO 单独检测到的目标提供受限的减速判断 |
| 构建脚本 | `SConstruct` 中注释掉了 `system/sensord`、`tools/webcam` 的 SConscript（原因见下） |
| 换行符 | `.gitattributes` 强制 `eol=lf`，全仓库已统一为 LF |

## 未随本分支迁移的模块

这些目录在 dragonpilot 中存在，但本仓库**没有**包含：

| 目录 | 原用途 | 现状 |
|---|---|---|
| `system/sensord/` | C++ 版 IMU / 磁力计驱动（bmx055、lsm6ds3、mmc5603nj） | 未迁移。Python 版 `system/sensordnew/` 保留 |
| `tools/webcam/` | USB 摄像头采集工具 | 未迁移 |
| `agent/` | 本地 AI agent（MCP server / DeepSeek 接入） | 未迁移 |
| `rear/` | 后视摄像头 Qt 程序 | 未迁移 |

因此 `SConstruct` 里对应的两行 `SConscript(...)` 已注释，否则 `scons` 会因为找不到文件直接失败。

## 已知问题

1. **sensord 运行时缺失**：`system/manager/process_config.py` 里仍然是

   ```python
   NativeProcess("sensord", "system/sensord", ["./sensord"], only_onroad),
   ```

   但 `system/sensord/` 目录不存在，这一进程会一直启动失败，陀螺仪 / 加速度计没有数据，
   依赖 IMU 的定位与校准会受影响。需要改成下面这行（`system/sensordnew` 是 Python 实现）：

   ```python
   PythonProcess("sensord", "system.sensordnew.sensord", only_onroad),
   ```

2. 未迁移模块中如果还有其它代码引用（例如 UI 里指向 `dragonpilot/agent` 的入口），
   在实际运行时才会暴露，本分支未做完整清理。

## 构建与运行

依赖与上游 openpilot 相同：Python 3.11 / 3.12、clang、OpenCL、Cap'n Proto 等，详见 `docs/`。

```bash
# Python 依赖，定义在 pyproject.toml 的 dependencies 里
pip install -e .

# 编译
scons -u -j$(nproc)

# 启动
./launch_openpilot.sh
```

`scons` 只编译当前分支实际存在的模块；因为 sensord / webcam 已从构建里摘掉，
首次编译不会再因为缺文件报错。

## 关于数据

本仓库**只包含源代码**，不包含任何行车数据：

- 没有摄像头画面、没有陀螺仪 / 加速度计原始数据、没有 route 日志
- `realdata/`、`*.hevc`、`fcamera.*`、`qcamera.*`、`rlog*` 等日志文件不应提交到本仓库
- 上游 openpilot 默认会把行车数据上传到 comma 的服务器；本分支不保证这一行为，
  是否上传取决于你自己的设备配置，请在设置里自行确认

## 许可与免责

openpilot 以 MIT 协议发布，部分组件使用其它协议，详见 `LICENSE` 与各目录下的许可文件。

**这是研究用途的 alpha 质量软件，不是产品。**
使用者需自行遵守当地法律法规，并对使用后果负责。无任何明示或暗示的担保。
