#include "selfdrive/ui/qt/onroad/model.h"

constexpr int CLIP_MARGIN = 500;
constexpr float MIN_DRAW_DISTANCE = 10.0;
constexpr float MAX_DRAW_DISTANCE = 100.0;

static int get_path_length_idx(const cereal::XYZTData::Reader &line, const float path_height) {
  const auto &line_x = line.getX();
  int max_idx = 0;
  for (int i = 1; i < line_x.size() && line_x[i] <= path_height; ++i) {
    max_idx = i;
  }
  return max_idx;
}

static bool interp_lane_y(const cereal::XYZTData::Reader &line, const float x, float *y) {
  const auto line_x = line.getX();
  const auto line_y = line.getY();
  if (line_x.size() == 0 || line_x.size() != line_y.size() || x < line_x[0] || x > *(line_x.end() - 1)) {
    return false;
  }

  int idx = 1;
  while (idx < line_x.size() && line_x[idx] < x) {
    ++idx;
  }

  if (idx == line_x.size()) {
    *y = line_y[idx - 1];
    return true;
  }

  const float x0 = line_x[idx - 1];
  const float x1 = line_x[idx];
  const float t = (x - x0) / std::max(x1 - x0, 1e-4f);
  *y = line_y[idx - 1] + t * (line_y[idx] - line_y[idx - 1]);
  return true;
}

void ModelRenderer::draw(QPainter &painter, const QRect &surface_rect) {
  auto *s = uiState();
  auto &sm = *(s->sm);
  // Check if data is up-to-date
  if (sm.rcv_frame("liveCalibration") < s->scene.started_frame ||
      sm.rcv_frame("modelV2") < s->scene.started_frame) {
    return;
  }

  clip_region = surface_rect.adjusted(-CLIP_MARGIN, -CLIP_MARGIN, CLIP_MARGIN, CLIP_MARGIN);
  experimental_mode = sm["selfdriveState"].getSelfdriveState().getExperimentalMode();
  longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  path_offset_z = sm["liveCalibration"].getLiveCalibration().getHeight()[0];

  painter.save();

  const auto &model = sm["modelV2"].getModelV2();
  const auto &radar_state = sm["radarState"].getRadarState();
  const auto &lead_one = radar_state.getLeadOne();

  update_model(model, lead_one);
  drawLaneLines(painter);
  drawPath(painter, model, surface_rect.height());

  if (longitudinal_control && sm.alive("radarState")) {
    update_leads(radar_state, model.getPosition());
    const auto &lead_two = radar_state.getLeadTwo();
    if (lead_one.getStatus()) {
      drawLead(painter, lead_one, lead_vertices[0], surface_rect);
      if (model.getYoloLead() && lead_one.getRadar()) {
        drawYoloRadarLead(painter, lead_one, lead_vertices[0], surface_rect);
      }
    }
    if (lead_two.getStatus() && (std::abs(lead_one.getDRel() - lead_two.getDRel()) > 3.0)) {
      drawLead(painter, lead_two, lead_vertices[1], surface_rect);
    }

    if (sm.alive("liveTracks")) {
      const auto &live_tracks = sm["liveTracks"].getLiveTracks();
      drawAdjacentLeadSpeeds(painter, live_tracks, model, surface_rect);
    }
  }

  if (s->scene.dp_ui_radar_tracks) {
    const auto &live_tracks = sm["liveTracks"].getLiveTracks();
    drawLiveTracks(painter, live_tracks, model, surface_rect);
  }

  painter.restore();
}

void ModelRenderer::update_leads(const cereal::RadarState::Reader &radar_state, const cereal::XYZTData::Reader &line) {
  for (int i = 0; i < 2; ++i) {
    if (i==1)
      continue;
    const auto &lead_data = (i == 0) ? radar_state.getLeadOne() : radar_state.getLeadTwo();
    if (lead_data.getStatus()) {
      float z = line.getZ()[get_path_length_idx(line, lead_data.getDRel())];
      mapToScreen(lead_data.getDRel(), -lead_data.getYRel(), z + path_offset_z, &lead_vertices[i]);
    }
  }
}

void ModelRenderer::update_model(const cereal::ModelDataV2::Reader &model, const cereal::RadarState::LeadData::Reader &lead) {
  const auto &model_position = model.getPosition();
  float max_distance = std::clamp(*(model_position.getX().end() - 1), MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE);

  // update lane lines
  const auto &lane_lines = model.getLaneLines();
  const auto &line_probs = model.getLaneLineProbs();
  int max_idx = get_path_length_idx(lane_lines[0], max_distance);
  for (int i = 0; i < std::size(lane_line_vertices); i++) {
    lane_line_probs[i] = line_probs[i];
    mapLineToPolygon(lane_lines[i], 0.025 * lane_line_probs[i], 0, &lane_line_vertices[i], max_idx);
  }

  // update road edges
  const auto &road_edges = model.getRoadEdges();
  const auto &edge_stds = model.getRoadEdgeStds();
  for (int i = 0; i < std::size(road_edge_vertices); i++) {
    road_edge_stds[i] = edge_stds[i];
    mapLineToPolygon(road_edges[i], 0.025, 0, &road_edge_vertices[i], max_idx);
  }

  // update path
  if (lead.getStatus()) {
    const float lead_d = lead.getDRel() * 2.;
    max_distance = std::clamp((float)(lead_d - fmin(lead_d * 0.35, 10.)), 0.0f, max_distance);
  }
  max_idx = get_path_length_idx(model_position, max_distance);
  mapLineToPolygon(model_position, 0.9, path_offset_z, &track_vertices, max_idx, false);
}

void ModelRenderer::drawLaneLines(QPainter &painter) {
  // lanelines
  for (int i = 0; i < std::size(lane_line_vertices); ++i) {
    painter.setBrush(QColor::fromRgbF(1.0, 1.0, 1.0, std::clamp<float>(lane_line_probs[i], 0.0, 0.7)));
    painter.drawPolygon(lane_line_vertices[i]);
  }

  // road edges
  for (int i = 0; i < std::size(road_edge_vertices); ++i) {
    painter.setBrush(QColor::fromRgbF(1.0, 0, 0, std::clamp<float>(1.0 - road_edge_stds[i], 0.0, 1.0)));
    painter.drawPolygon(road_edge_vertices[i]);
  }
}

void ModelRenderer::drawPath(QPainter &painter, const cereal::ModelDataV2::Reader &model, int height) {
  QLinearGradient bg(0, height, 0, 0);

  auto *s = uiState();
  if (s->scene.dp_ui_rainbow) {
    constexpr int NUM_COLORS = 25;
    constexpr int ALPHA = 128;

    float v_ego = (*uiState()->sm)["carState"].getCarState().getVEgo();

    if (!dp_rainbow_init) {
      dp_rainbow_color_list.reserve(NUM_COLORS);
      for (int i = 0; i < NUM_COLORS; ++i) {
        qreal t = static_cast<qreal>(i) / (NUM_COLORS - 1);
        dp_rainbow_color_list.append(QColor::fromHsvF(t, 1.0, 1.0, ALPHA / 255.0));
      }
      dp_rainbow_init = true;
    }
    bg.setSpread(QGradient::RepeatSpread);
    // bigger = faster, however it is still limited to the global UI_FREQ (refresh rate)
    // only way to make it move faster is to reduce NUM_COLORS, but that will also reduce the color smoothness.
    qreal rotation_speed = std::max(0.01f, v_ego) / UI_FREQ;
    dp_rainbow_rotation -= rotation_speed;

    if (dp_rainbow_rotation < 0.0) {
      dp_rainbow_rotation += 1.0;
      dp_rainbow_color_list.append(dp_rainbow_color_list.takeFirst());
    }
    // fill color
    const qreal step = 1.0 / (NUM_COLORS - 1);
    for (int i = 0; i < NUM_COLORS; ++i) {
      bg.setColorAt(i * step, dp_rainbow_color_list.at(i));
    }

  } else if (experimental_mode) {
    // The first half of track_vertices are the points for the right side of the path
    const auto &acceleration = model.getAcceleration().getX();
    const int max_len = std::min<int>(track_vertices.length() / 2, acceleration.size());

    for (int i = 0; i < max_len; ++i) {
      // Some points are out of frame
      int track_idx = max_len - i - 1;  // flip idx to start from bottom right
      if (track_vertices[track_idx].y() < 0 || track_vertices[track_idx].y() > height) continue;

      // Flip so 0 is bottom of frame
      float lin_grad_point = (height - track_vertices[track_idx].y()) / height;

      // speed up: 120, slow down: 0
      float path_hue = fmax(fmin(60 + acceleration[i] * 35, 120), 0);
      // FIXME: painter.drawPolygon can be slow if hue is not rounded
      path_hue = int(path_hue * 100 + 0.5) / 100;

      float saturation = fmin(fabs(acceleration[i] * 1.5), 1);
      float lightness = util::map_val(saturation, 0.0f, 1.0f, 0.95f, 0.62f);        // lighter when grey
      float alpha = util::map_val(lin_grad_point, 0.75f / 2.f, 0.75f, 0.4f, 0.0f);  // matches previous alpha fade
      bg.setColorAt(lin_grad_point, QColor::fromHslF(path_hue / 360., saturation, lightness, alpha));

      // Skip a point, unless next is last
      i += (i + 2) < max_len ? 1 : 0;
    }

  } else {
    updatePathGradient(bg);
  }

  painter.setBrush(bg);
  painter.drawPolygon(track_vertices);
}

void ModelRenderer::updatePathGradient(QLinearGradient &bg) {
  static const QColor throttle_colors[] = {
      QColor::fromHslF(148. / 360., 0.94, 0.51, 0.4),
      QColor::fromHslF(112. / 360., 1.0, 0.68, 0.35),
      QColor::fromHslF(112. / 360., 1.0, 0.68, 0.0)};

  static const QColor no_throttle_colors[] = {
      QColor::fromHslF(148. / 360., 0.0, 0.95, 0.4),
      QColor::fromHslF(112. / 360., 0.0, 0.95, 0.35),
      QColor::fromHslF(112. / 360., 0.0, 0.95, 0.0),
  };

  // Transition speed; 0.1 corresponds to 0.5 seconds at UI_FREQ
  constexpr float transition_speed = 0.1f;

  // Start transition if throttle state changes
  bool allow_throttle = (*uiState()->sm)["longitudinalPlan"].getLongitudinalPlan().getAllowThrottle() || !longitudinal_control;
  if (allow_throttle != prev_allow_throttle) {
    prev_allow_throttle = allow_throttle;
    // Invert blend factor for a smooth transition when the state changes mid-animation
    blend_factor = std::max(1.0f - blend_factor, 0.0f);
  }

  const QColor *begin_colors = allow_throttle ? no_throttle_colors : throttle_colors;
  const QColor *end_colors = allow_throttle ? throttle_colors : no_throttle_colors;
  if (blend_factor < 1.0f) {
    blend_factor = std::min(blend_factor + transition_speed, 1.0f);
  }

  // Set gradient colors by blending the start and end colors
  bg.setColorAt(0.0f, blendColors(begin_colors[0], end_colors[0], blend_factor));
  bg.setColorAt(0.5f, blendColors(begin_colors[1], end_colors[1], blend_factor));
  bg.setColorAt(1.0f, blendColors(begin_colors[2], end_colors[2], blend_factor));
}

QColor ModelRenderer::blendColors(const QColor &start, const QColor &end, float t) {
  if (t == 1.0f) return end;
  return QColor::fromRgbF(
      (1 - t) * start.redF() + t * end.redF(),
      (1 - t) * start.greenF() + t * end.greenF(),
      (1 - t) * start.blueF() + t * end.blueF(),
      (1 - t) * start.alphaF() + t * end.alphaF());
}

void ModelRenderer::drawLiveTracks(QPainter &painter,
  const cereal::RadarData::Reader &live_tracks,
  const cereal::ModelDataV2::Reader &model_data,
  const QRect &surface_rect) {

  // Get the model's predicted path for Z-coordinate calculation
  const auto& model_path_position = model_data.getPosition();

  // Clear previously drawn positions for this frame
  drawn_track_positions.clear();

  // Set text properties
  painter.setPen(Qt::white);
  painter.setFont(QFont("Inter", 24, QFont::Bold));

  // 100x100 pixel deduplication region
  constexpr float REGION_W = 160.0f;
  constexpr float REGION_H = 110.0f;
  constexpr float HALF_W = REGION_W / 2.0f;
  constexpr float HALF_H = REGION_H / 2.0f;

  // Iterate through each radar point from live_tracks
  for (const auto& point : live_tracks.getPoints()) {
    float dRel = point.getDRel();
    float yRel = point.getYRel();
    // float yvRel = point.getYvRel();
    float vRel = point.getVRel();

    // Calculate Z-coordinate using the model's path
    float z_on_path = path_offset_z; // Default base offset

    // Ensure dRel is non-negative for indexing
    if (dRel >= 0) {
      z_on_path += model_path_position.getZ()[get_path_length_idx(model_path_position, dRel)];
    }

    QPointF screen_pos;
    // mapToScreen projects a point from car space to screen space
    if (mapToScreen(dRel, -yRel, z_on_path, &screen_pos)) { // yRel is negated as in update_leads
      // Check if any already-drawn point is within the 100x100 region
      bool too_close = false;
      for (const auto& drawn_pos : drawn_track_positions) {
        float dx = screen_pos.x() - drawn_pos.x();
        float dy = screen_pos.y() - drawn_pos.y();
        if (std::abs(dx) < HALF_W && std::abs(dy) < HALF_H) {
          too_close = true;
          break;
        }
      }
      if (too_close) continue;

      // Save this position for deduplication
      drawn_track_positions.append(screen_pos);

      // Basic drawing: Draw a small circle for the point
      painter.setBrush(QColor(255, 0, 0, 200)); // Red color for live tracks
      painter.drawEllipse(screen_pos, 10, 10); // Draw a small circle of radius 5

      // Prepare text to display
      // QString infoText = QString("ID: %1\nd: %2 m\ny: %3 m\ndV: %4 m/s\nyV: %5 m/s")
      //                      .arg(point.getTrackId())
      //                      .arg(dRel, 0, 'f', 2)
      //                      .arg(yRel, 0, 'f', 2)
      //                      .arg(vRel, 0, 'f', 2)
      //                      .arg(yvRel, 0, 'f', 2);
      QString infoText = QString("%2 \n %4 ")
                           .arg(dRel, 0, 'f', 1)
                           .arg(vRel*3.6, 0, 'f', 0);

      // Draw text near the point
      // Adjust text position for better visibility (e.g., slightly offset from the point)
      QRectF textRect(screen_pos.x() + 10, screen_pos.y() - 20, 260, 250); // Adjust size as needed
      painter.drawText(textRect, Qt::AlignLeft, infoText);
    }
  }
}

void ModelRenderer::drawAdjacentLeadSpeeds(QPainter &painter, const cereal::RadarData::Reader &live_tracks,
                                           const cereal::ModelDataV2::Reader &model_data, const QRect &surface_rect) {
  constexpr float LANE_LINE_PROB_THRESHOLD = 0.5f;
  constexpr float NOMINAL_LANE_WIDTH = 3.1f;
  constexpr float RADAR_TO_CAMERA = 1.52f;

  const auto &lane_lines = model_data.getLaneLines();
  const auto &model_lane_line_probs = model_data.getLaneLineProbs();
  if (lane_lines.size() < 4 || model_lane_line_probs.size() < 4) {
    return;
  }

  struct AdjacentLead {
    bool valid = false;
    bool left_lane = false;
    float d_rel = 0.0f;
    float y_rel = 0.0f;
    float v_rel = 0.0f;
  };

  AdjacentLead left_lead;
  AdjacentLead right_lead;

  for (const auto &point : live_tracks.getPoints()) {
    const float d_rel = point.getDRel();
    const float y_rel = point.getYRel();

    if (d_rel <= 0.0f || d_rel > MAX_DRAW_DISTANCE) {
      continue;
    }

    const float track_x = d_rel + RADAR_TO_CAMERA;
    float left_outer_y = 0.0f;
    float left_inner_y = 0.0f;
    float right_inner_y = 0.0f;
    float right_outer_y = 0.0f;
    const bool has_left_inner = model_lane_line_probs[1] >= LANE_LINE_PROB_THRESHOLD &&
                                interp_lane_y(lane_lines[1], track_x, &left_inner_y);
    const bool has_right_inner = model_lane_line_probs[2] >= LANE_LINE_PROB_THRESHOLD &&
                                 interp_lane_y(lane_lines[2], track_x, &right_inner_y);
    bool has_left_outer = model_lane_line_probs[0] >= LANE_LINE_PROB_THRESHOLD &&
                          interp_lane_y(lane_lines[0], track_x, &left_outer_y);
    bool has_right_outer = model_lane_line_probs[3] >= LANE_LINE_PROB_THRESHOLD &&
                           interp_lane_y(lane_lines[3], track_x, &right_outer_y);

    if (!has_left_outer && has_left_inner) {
      left_outer_y = left_inner_y + NOMINAL_LANE_WIDTH;
      has_left_outer = true;
    }
    if (!has_right_outer && has_right_inner) {
      right_outer_y = right_inner_y - NOMINAL_LANE_WIDTH;
      has_right_outer = true;
    }

    // Keep the same lateral convention as lane_centering_curvature_offset:
    // model lane lines are negated before comparing against radar yRel.
    bool in_left_lane = false;
    bool in_current_lane = false;
    bool in_right_lane = false;
    if (has_left_inner && has_left_outer) {
      const float left_lane_min = std::min(-left_outer_y, -left_inner_y);
      const float left_lane_max = std::max(-left_outer_y, -left_inner_y);
      const float lane_width = left_lane_max - left_lane_min;
      const float lane_margin = std::min(0.35f, lane_width * 0.2f);
      in_left_lane = lane_width >= 2.0f && lane_width <= 5.0f &&
                     y_rel >= left_lane_min + lane_margin && y_rel <= left_lane_max - lane_margin;
    }

    if (has_left_inner && has_right_inner) {
      const float current_lane_min = std::min(-left_inner_y, -right_inner_y);
      const float current_lane_max = std::max(-left_inner_y, -right_inner_y);
      const float lane_width = current_lane_max - current_lane_min;
      const float lane_margin = std::min(0.35f, lane_width * 0.2f);
      in_current_lane = lane_width >= 2.0f && lane_width <= 5.0f &&
                        y_rel >= current_lane_min + lane_margin &&
                        y_rel <= current_lane_max - lane_margin;
    }

    if (has_right_inner && has_right_outer) {
      const float right_lane_min = std::min(-right_inner_y, -right_outer_y);
      const float right_lane_max = std::max(-right_inner_y, -right_outer_y);
      const float lane_width = right_lane_max - right_lane_min;
      const float lane_margin = std::min(0.35f, lane_width * 0.2f);
      in_right_lane = lane_width >= 2.0f && lane_width <= 5.0f &&
                      y_rel >= right_lane_min + lane_margin && y_rel <= right_lane_max - lane_margin;
    }

    // The current lane has priority at the boundaries. Never display its
    // lead speed; only targets in an adjacent lane are shown.
    if (in_current_lane || (!in_left_lane && !in_right_lane)) {
      continue;
    }

    auto &lead = in_left_lane ? left_lead : right_lead;
    if (!lead.valid || d_rel < lead.d_rel) {
      lead = {true, in_left_lane, d_rel, y_rel, point.getVRel()};
    }
  }

  painter.save();
  painter.setPen(Qt::white);
  painter.setFont(QFont("Inter", 30, QFont::Bold));

  const float v_ego = (*uiState()->sm)["carState"].getCarState().getVEgo();
  for (const auto &lead : {left_lead, right_lead}) {
    if (!lead.valid) {
      continue;
    }

    const float d_rel = lead.d_rel;
    const float y_rel = lead.y_rel;
    const auto &line = model_data.getPosition();
    const float z = line.getZ()[get_path_length_idx(line, d_rel)];
    QPointF lead_pos;
    if (!mapToScreen(d_rel, -y_rel, z + path_offset_z, &lead_pos)) {
      continue;
    }

    float v_lead_kph = (v_ego + lead.v_rel) * 3.6f;
    if (!lead.left_lane) {
      v_lead_kph = std::max(0.0f, v_lead_kph);
    }
    const QString speed_text = QString("%1 ").arg(v_lead_kph, 0, 'f', 0);
    const float x = std::clamp<float>(lead_pos.x(), 0.f, surface_rect.width());
    const float y = std::clamp<float>(lead_pos.y(), 0.f, surface_rect.height());
    const float text_offset_x = y_rel > 0.0f ? -110.0f : 30.0f;
    painter.drawText(QPointF(x + text_offset_x, y - 35.0f), speed_text);
  }

  painter.restore();
}

void ModelRenderer::drawLead(QPainter &painter, const cereal::RadarState::LeadData::Reader &lead_data,
                             const QPointF &vd, const QRect &surface_rect) {
  const float speedBuff = 10.;
  const float leadBuff = 40.;
  const float d_rel = lead_data.getDRel();
  const float v_rel = lead_data.getVRel();

  float fillAlpha = 0;
  if (d_rel < leadBuff) {
    fillAlpha = 255 * (1.0 - (d_rel / leadBuff));
    if (v_rel < 0) {
      fillAlpha += 255 * (-1 * (v_rel / speedBuff));
    }
    fillAlpha = (int)(fmin(fillAlpha, 255));
  }

  float sz = std::clamp((25 * 30) / (d_rel / 3 + 30), 15.0f, 30.0f) * 2.35;
  float x = std::clamp<float>(vd.x(), 0.f, surface_rect.width() - sz / 2);
  float y = std::min<float>(vd.y(), surface_rect.height() - sz * 0.6);

  float g_xo = sz / 5;
  float g_yo = sz / 10;

  QPointF glow[] = {{x + (sz * 1.35) + g_xo, y + sz + g_yo}, {x, y - g_yo}, {x - (sz * 1.35) - g_xo, y + sz + g_yo}};
  painter.setBrush(QColor(218, 202, 37, 255));
  painter.drawPolygon(glow, std::size(glow));

  // chevron
  QPointF chevron[] = {{x + (sz * 1.25), y + sz}, {x, y}, {x - (sz * 1.25), y + sz}};
  painter.setBrush(QColor(201, 34, 49, fillAlpha));
  painter.drawPolygon(chevron, std::size(chevron));

  // Draw distance and speed text near the lead indicator
  painter.save();

  if (d_rel > 27.0) {
    painter.setPen(Qt::green);
  } else {
    painter.setPen(Qt::red);
  }
  painter.setFont(QFont("Inter", 30, QFont::Bold));
  QString distanceText = QString("%1 m").arg(d_rel, 0, 'f', 1);
  QPointF distancePos(x, y - sz * 0.4);
  painter.drawText(distancePos, distanceText);

  painter.setFont(QFont("Inter", 25, QFont::Bold));
  float v_ego = (*uiState()->sm)["carState"].getCarState().getVEgo();
  float v_lead_kph = (v_ego + v_rel) * 3.6f;
  if (v_lead_kph < 0.0f) {
    v_lead_kph = 0.0f;
  }
  QString speedText = QString("%1 ").arg(v_lead_kph, 0, 'f', 0);
  QPointF speedPos(x-199, y - sz * 0.95);
  painter.drawText(speedPos, speedText);
  painter.restore();
}

void ModelRenderer::drawYoloRadarLead(QPainter &painter, const cereal::RadarState::LeadData::Reader &lead_data,
                                      const QPointF &vd, const QRect &surface_rect) {
  const float d_rel = lead_data.getDRel();
  const float size = std::clamp(2600.0f / (d_rel + 20.0f), 42.0f, 105.0f);
  const float x = std::clamp<float>(vd.x(), size / 2.0f, surface_rect.width() - size / 2.0f);
  const float y = std::clamp<float>(vd.y(), size / 2.0f, surface_rect.height() - size / 2.0f);

  painter.save();
  painter.setBrush(Qt::NoBrush);
  painter.setPen(QPen(QColor(53, 224, 115), 4.0f));
  painter.drawRect(QRectF(x - size / 2.0f, y - size / 2.0f, size, size));
  painter.restore();
}

// Projects a point in car to space to the corresponding point in full frame image space.
bool ModelRenderer::mapToScreen(float in_x, float in_y, float in_z, QPointF *out) {
  Eigen::Vector3f input(in_x, in_y, in_z);
  auto pt = car_space_transform * input;
  *out = QPointF(pt.x() / pt.z(), pt.y() / pt.z());
  return clip_region.contains(*out);
}

void ModelRenderer::mapLineToPolygon(const cereal::XYZTData::Reader &line, float y_off, float z_off,
                                     QPolygonF *pvd, int max_idx, bool allow_invert) {
  const auto line_x = line.getX(), line_y = line.getY(), line_z = line.getZ();
  QPointF left, right;
  pvd->clear();
  for (int i = 0; i <= max_idx; i++) {
    // highly negative x positions  are drawn above the frame and cause flickering, clip to zy plane of camera
    if (line_x[i] < 0) continue;

    bool l = mapToScreen(line_x[i], line_y[i] - y_off, line_z[i] + z_off, &left);
    bool r = mapToScreen(line_x[i], line_y[i] + y_off, line_z[i] + z_off, &right);
    if (l && r) {
      // For wider lines the drawn polygon will "invert" when going over a hill and cause artifacts
      if (!allow_invert && pvd->size() && left.y() > pvd->back().y()) {
        continue;
      }
      pvd->push_back(left);
      pvd->push_front(right);
    }
  }
}
