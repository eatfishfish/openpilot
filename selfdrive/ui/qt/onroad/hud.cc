#include "selfdrive/ui/qt/onroad/hud.h"

#include <cmath>

#include <QJsonDocument>
#include <QJsonObject>

#include "common/params.h"
#include "selfdrive/ui/qt/util.h"

constexpr int SET_SPEED_NA = 255;

static QString turnIconLabel(int turn_icon) {
  switch (turn_icon) {
    case 2:
      return QStringLiteral("↰");
    case 3:
      return QStringLiteral("↱");
    case 4:
      return QStringLiteral("↖");
    case 5:
      return QStringLiteral("↗");
    case 8:
      return QStringLiteral("↶");
    default:
      return QStringLiteral("↑");
  }
}

HudRenderer::HudRenderer() {}

void HudRenderer::updateState(const UIState &s) {
  is_metric = s.scene.is_metric;
  status = s.status;

  if (!amap_read_timer.isValid() || amap_read_timer.hasExpired(200)) {
    const std::string amap_info = Params().get("AmapInfo");
    if (!amap_info.empty()) {
      const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(amap_info));
      if (doc.isObject()) {
        const QJsonObject obj = doc.object();
        amap_road_name = obj.value("CUR_ROAD_NAME").toString();
        amap_next_road_name = obj.value("NEXT_ROAD_NAME").toString();
        amap_turn_text = obj.value("TURN_TEXT").toString();
        amap_speed_limit_source = obj.value("SPEED_LIMIT_SOURCE").toString();
        amap_speed_limit = obj.value("LIMITED_SPEED").toInt();
        amap_next_distance = obj.value("SEG_REMAIN_DIS").toInt();
        amap_turn_icon = obj.value("NEW_ICON").toInt();
        amap_distance_day_m = obj.value("DISTANCE_DAY_M").toDouble();
        amap_distance_month_m = obj.value("DISTANCE_MONTH_M").toDouble();
        has_amap_info = !amap_next_road_name.isEmpty() || !amap_road_name.isEmpty() ||
                        amap_speed_limit > 0 || amap_next_distance > 0;
      }
    }
    amap_read_timer.start();
  }

  const SubMaster &sm = *(s.sm);
  if (sm.rcv_frame("carState") < s.scene.started_frame) {
    is_cruise_set = false;
    set_speed = SET_SPEED_NA;
    speed = 0.0;
    return;
  }

  const auto &controls_state = sm["controlsState"].getControlsState();
  const auto &car_state = sm["carState"].getCarState();

  // Handle older routes where vCruiseCluster is not set
  set_speed = car_state.getVCruiseCluster() == 0.0 ? controls_state.getVCruiseDEPRECATED() : car_state.getVCruiseCluster();
  is_cruise_set = set_speed > 0 && set_speed != SET_SPEED_NA;
  is_cruise_available = set_speed != -1;

  if (is_cruise_set && !is_metric) {
    set_speed *= KM_TO_MILE;
  }

  // Handle older routes where vEgoCluster is not set
  v_ego_cluster_seen = v_ego_cluster_seen || car_state.getVEgoCluster() != 0.0;
  float v_ego = v_ego_cluster_seen ? car_state.getVEgoCluster() : car_state.getVEgo();
  speed = std::max<float>(0.0f, v_ego * (is_metric ? MS_TO_KPH : MS_TO_MPH));
}

void HudRenderer::draw(QPainter &p, const QRect &surface_rect) {
  p.save();

  // Draw header gradient
  QLinearGradient bg(0, UI_HEADER_HEIGHT - (UI_HEADER_HEIGHT / 2.5), 0, UI_HEADER_HEIGHT);
  bg.setColorAt(0, QColor::fromRgbF(0, 0, 0, 0.45));
  bg.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0));
  p.fillRect(0, 0, surface_rect.width(), UI_HEADER_HEIGHT, bg);


  if (is_cruise_available) {
    drawSetSpeed(p, surface_rect);
  }
  drawCurrentSpeed(p, surface_rect);
  drawAmapInfo(p, surface_rect);

  p.restore();
}

void HudRenderer::drawSetSpeed(QPainter &p, const QRect &surface_rect) {
  // Draw outer box + border to contain set speed
  const QSize default_size = {172, 204};
  QSize set_speed_size = is_metric ? QSize(200, 204) : default_size;
  QRect set_speed_rect(QPoint(60 + (default_size.width() - set_speed_size.width()) / 2, 45), set_speed_size);

  // Draw set speed box
  p.setPen(QPen(QColor(255, 255, 255, 75), 6));
  p.setBrush(QColor(0, 0, 0, 166));
  p.drawRoundedRect(set_speed_rect, 32, 32);

  // Colors based on status
  QColor max_color = QColor(0xa6, 0xa6, 0xa6, 0xff);
  QColor set_speed_color = QColor(0x72, 0x72, 0x72, 0xff);
  if (is_cruise_set) {
    set_speed_color = QColor(255, 255, 255);
    if (status == STATUS_DISENGAGED) {
      max_color = QColor(255, 255, 255);
    } else if (status == STATUS_OVERRIDE) {
      max_color = QColor(0x91, 0x9b, 0x95, 0xff);
    } else {
      max_color = QColor(0x80, 0xd8, 0xa6, 0xff);
    }
  }

  // Draw "MAX" text
  p.setFont(InterFont(40, QFont::DemiBold));
  p.setPen(max_color);
  p.drawText(set_speed_rect.adjusted(0, 27, 0, 0), Qt::AlignTop | Qt::AlignHCenter, tr("MAX"));

  // Draw set speed
  QString setSpeedStr = is_cruise_set ? QString::number(std::nearbyint(set_speed)) : "–";
  p.setFont(InterFont(90, QFont::Bold));
  p.setPen(set_speed_color);
  p.drawText(set_speed_rect.adjusted(0, 77, 0, 0), Qt::AlignTop | Qt::AlignHCenter, setSpeedStr);
}

void HudRenderer::drawCurrentSpeed(QPainter &p, const QRect &surface_rect) {
  QString speedStr = QString::number(std::nearbyint(speed));

  p.setFont(InterFont(176, QFont::Bold));
  drawText(p, surface_rect.center().x(), 210, speedStr);

  p.setFont(InterFont(66));
  drawText(p, surface_rect.center().x(), 290, is_metric ? tr("km/h") : tr("mph"), 200);
}

void HudRenderer::drawAmapInfo(QPainter &p, const QRect &surface_rect) {
  if (!has_amap_info) {
    return;
  }

  const QRect card(surface_rect.width()-370, surface_rect.height() - 200, 370, 210);
  p.setPen(QPen(QColor(255, 255, 255, 70), 4));
  p.setBrush(QColor(0, 0, 0, 150));
  p.drawRoundedRect(card, 28, 28);

  p.setPen(QColor(255, 255, 255, 235));
  p.setFont(InterFont(34, QFont::DemiBold));
  const QString display_road_name = !amap_next_road_name.isEmpty() ? amap_next_road_name : amap_road_name;
  p.drawText(card.adjusted(24, 16, -24, 0), Qt::AlignLeft | Qt::TextSingleLine,
             display_road_name.isEmpty() ? tr("未知道路") : display_road_name);

  const QString limit = amap_speed_limit > 0 ? QString::number(amap_speed_limit) + " km/h" : "–";
  const QString source = amap_speed_limit_source.contains("inferred") ? tr("推测") : tr("高德");
  p.setFont(InterFont(30, QFont::DemiBold));
  p.setPen(QColor(0x80, 0xd8, 0xa6));
  p.drawText(card.adjusted(24, 62, -24, 0), Qt::AlignLeft | Qt::TextSingleLine, tr("限速 ") + limit + " " + source);

  p.setPen(QColor(255, 255, 255, 235));
  p.setFont(InterFont(104, QFont::Bold));
  p.drawText(card.adjusted(14, 60, -248, 0), Qt::AlignCenter | Qt::TextSingleLine, turnIconLabel(amap_turn_icon));

  p.setPen(QColor(255, 255, 255, 210));
  p.setFont(InterFont(36, QFont::Bold));
  p.drawText(card.adjusted(124, 100, -24, 0), Qt::AlignLeft | Qt::TextSingleLine,
             QString::number(amap_next_distance) + " m");

  p.setFont(InterFont(25));
  p.setPen(QColor(255, 255, 255, 180));
  const QString distance = tr("今日 ") + QString::number(amap_distance_day_m / 1000.0, 'f', 1) + " km  " +
                           tr("本月 ") + QString::number(amap_distance_month_m / 1000.0, 'f', 1) + " km";
  p.drawText(card.adjusted(24, 150, -24, 0), Qt::AlignLeft | Qt::TextSingleLine, distance);
}

void HudRenderer::drawText(QPainter &p, int x, int y, const QString &text, int alpha) {
  QRect real_rect = p.fontMetrics().boundingRect(text);
  real_rect.moveCenter({x, y - real_rect.height() / 2});

  p.setPen(QColor(0xff, 0xff, 0xff, alpha));
  p.drawText(real_rect.x(), real_rect.bottom(), text);
}
