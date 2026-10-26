# YOLO Vision-Only 前车兜底方案

## 目标

当满足以下条件时，为 YOLO 提供受限的 vision-only 前车兜底：

```text
modelV2 没有有效前车
YOLO 高置信度检测到本车道内障碍物
雷达暂时没有匹配到同一目标
```

该方案用于停止加速、收油和弱减速的辅助判断。不能仅依赖 YOLO vision-only 触发强制动、FCW 或高速静止目标制动。

## 数据来源优先级

```text
YOLO + 雷达匹配：使用雷达距离、速度、加速度
modelV2 + 雷达匹配：使用雷达距离、速度、加速度
modelV2 vision-only：保留现有逻辑
YOLO vision-only：受限兜底，不覆盖已有有效 modelV2 lead
```

YOLO 单目距离由目标框宽度、目标真实宽度假设和相机焦距计算，不能作为雷达距离的直接替代。

## 第一层：YOLO 稳定候选

在 `selfdrive/modeld/yolo.py` 中维护稳定候选状态，而不是使用单帧检测结果。

建议最小准入条件：

```text
YOLO 置信度 >= 0.70
在 modeld 当前车道线内
左右车道线置信度足够
连续检测 >= 3 帧
距离处于合理范围，例如 5m 到 100m
相邻帧距离变化 <= max(2m, 距离 * 0.15)
相邻帧横向变化 <= 0.4m
```

建议维护的数据：

```text
最近 x/y
最近检测时间
连续检测帧数
距离变化率
横向变化率
候选类别和置信度
```

YOLO 没有可靠单目速度，不能只凭单帧或距离变化把目标直接判为静止车辆。

## 第二层：modeld 发布来源标记

在 `selfdrive/modeld/modeld.py` 中，仅在 modelV2 没有有效前车时发布 YOLO vision-only 候选：

```text
modelV2 lead_prob <= 0.5
YOLO 稳定候选有效
```

需要区分两种 YOLO 结果：

```text
yoloLead = true
yoloVisionOnly = false：YOLO 已被雷达确认

yoloLead = true
yoloVisionOnly = true：YOLO 稳定候选，但未被雷达确认
```

建议在 `cereal/log.capnp` 的 `ModelDataV2` 中增加 `yoloVisionOnly` 字段，避免仅靠 `yoloLead` 产生歧义。

发布给 `leadsV3[0]` 的基础数据：

```text
x = yolo_x
y = yolo_y
prob = yolo_prob
```

## 第三层：radard vision-only fallback

当前 `selfdrive/controls/radard.py` 对 YOLO lead 的原则是必须雷达匹配。扩展后应保持默认严格模式：

```text
YOLO + 雷达匹配：
  Track.get_RadarState()
  使用雷达 dRel、vLead、aLeadK

YOLO vision-only：
  仅当 yoloVisionOnly = true 时构造非雷达 lead
```

建议的 YOLO vision-only `RadarState.LeadData`：

```text
dRel = yolo_x - RADAR_TO_CAMERA
yRel = -yolo_y
vRel = 0
vLead = v_ego
vLeadK = v_ego
aLeadK = 0
aLeadTau = 0.3
modelProb = yolo_prob
radar = false
radarTrackId = -1
```

`vLead = v_ego` 表示目标速度未知时的保守中性假设：不把目标直接当成静止车，也不据此产生激烈制动。

## 第四层：纵向控制限制

MPC 不能把 YOLO vision-only 当成完整雷达 lead 使用。需要将 lead 来源传递到纵向规划层，例如：

```text
leadSource = radar
leadSource = modelVision
leadSource = yoloVision
```

对于 `yoloVision`：

```text
允许停止继续加速、收油和弱减速
限制最大减速度，例如 -0.8 到 -1.2 m/s^2
禁止由该来源单独触发 FCW
禁止由该来源单独触发强制动
连续丢失 2 到 3 帧后失效
```

不要在 `long_mpc.py` 修改全局 `ACCEL_MIN`。应只对 `yoloVision` 来源增加局部约束，避免影响雷达确认前车。

## 来源切换

必须避免来源切换造成距离跳变，例如：

```text
YOLO vision-only：4.3m
雷达匹配后：6.1m
```

建议规则：

```text
无 lead -> YOLO vision-only：先收油或弱减速
YOLO vision-only -> 雷达匹配：平滑 dRel，切换至雷达 v/a
雷达 lead 丢失但 YOLO 仍稳定：最多保留 0.2 到 0.3 秒
YOLO 连续丢失：立即失效，不长期保留假目标
```

雷达匹配成功后始终以雷达距离、速度、加速度为准。

## UI 和日志

UI 应明确区分来源：

```text
YOLO + 雷达确认：绿色实线矩形
YOLO vision-only：黄色虚线矩形
普通前车：现有箭头和距离/速度显示
```

统计日志增加：

```text
yolo_vision_only_candidate_count
yolo_vision_only_accepted_count
候选连续稳定帧数
首次检测到首次发布 lead 的延迟
转为雷达匹配的延迟
与后续雷达距离的误差
vision-only 持续时长
被拒绝的原因
```

## 推荐实现顺序

1. 增加稳定候选状态机、来源标记和日志，不参与控制。
2. 发布为 `radarState` lead 并以 UI 显示，但不让 MPC 使用。
3. 路测统计误检率、连续性和后续雷达匹配误差。
4. 启用停止加速、收油和弱减速。
5. 仅在充分验证后评估进一步控制权限。

## 安全边界

```text
不允许 YOLO vision-only 覆盖有效 modelV2 lead。
不允许 YOLO vision-only 绕过当前车道线判断。
不允许 YOLO vision-only 绕过稳定性判断。
不允许 YOLO vision-only 单独触发高速强制动。
不允许放宽雷达匹配条件来伪造“已确认”状态。
```
