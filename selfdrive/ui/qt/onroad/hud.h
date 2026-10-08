#pragma once

#include <QElapsedTimer>
#include <QJsonObject>
#include <QPainter>
#include "selfdrive/ui/ui.h"

class HudRenderer : public QObject {
  Q_OBJECT

public:
  HudRenderer();
  void updateState(const UIState &s);
  void draw(QPainter &p, const QRect &surface_rect);

private:
  void drawSetSpeed(QPainter &p, const QRect &surface_rect);
  void drawCurrentSpeed(QPainter &p, const QRect &surface_rect);
  void drawAmapInfo(QPainter &p, const QRect &surface_rect);
  void drawText(QPainter &p, int x, int y, const QString &text, int alpha = 255);

  float speed = 0;
  float set_speed = 0;
  int amap_speed_limit = 0;
  int amap_next_distance = 0;
  int amap_turn_icon = 0;
  double amap_distance_day_m = 0.0;
  double amap_distance_month_m = 0.0;
  QString amap_road_name;
  QString amap_next_road_name;
  QString amap_turn_text;
  QString amap_speed_limit_source;
  bool is_cruise_set = false;
  bool is_cruise_available = true;
  bool has_amap_info = false;
  bool is_metric = false;
  bool v_ego_cluster_seen = false;
  int status = STATUS_DISENGAGED;
  QElapsedTimer amap_read_timer;
};
