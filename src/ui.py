"""RainCast AI — Modern PyQt6 Rain Prediction & Live Turkey Weather Map Desktop App."""

from __future__ import annotations

import datetime
import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Allow running both as `python app.py` and `python src/ui.py`
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import FEATURE_NAMES, build_single_feature_row  # noqa: E402
from weather_api import (  # noqa: E402
    TURKEY_CITIES,
    fetch_city_forecast,
    fetch_turkey_map_weather,
    resolve_city_name,
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_FEATURES = FEATURE_NAMES

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

STYLESHEET = """
QWidget {
    background-color: #0B1120;
    color: #F8FAFC;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

QFrame#TopBar {
    background-color: #0F172A;
    border-bottom: 1px solid #1E293B;
}

QFrame#Card {
    background-color: #131C31;
    border: 1px solid #1E2D4A;
    border-radius: 16px;
}

QFrame#HeroCard {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14213D, stop:1 #111827);
    border: 1px solid #263A5E;
    border-radius: 16px;
}

QFrame#MetricTile {
    background-color: #0E1629;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
}

QFrame#DayCard {
    background-color: #0E1629;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
}

QFrame#StatPill {
    background-color: #131C31;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
}

QFrame#NavContainer {
    background-color: #0B1120;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
}

QLabel {
    background-color: transparent;
}

QLabel#AppTitle {
    font-size: 17px;
    font-weight: 700;
    color: #F8FAFC;
}

QLabel#AppSubtitle {
    font-size: 11px;
    color: #64748B;
}

QLabel#SectionTitle {
    font-size: 15px;
    font-weight: 700;
    color: #F1F5F9;
}

QLabel#FieldLabel {
    font-size: 12px;
    font-weight: 600;
    color: #94A3B8;
}

QLabel#MutedText {
    font-size: 12px;
    color: #64748B;
}

QLabel#VerdictTitle {
    font-size: 22px;
    font-weight: 700;
    color: #F8FAFC;
}

QLabel#ProbabilityBig {
    font-size: 32px;
    font-weight: 800;
    color: #38BDF8;
}

QLabel#RiskBadge {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
    background-color: #1E293B;
    color: #94A3B8;
}

QLabel#MetricValue {
    font-size: 15px;
    font-weight: 700;
    color: #F8FAFC;
}

QLabel#MetricTitle {
    font-size: 11px;
    color: #64748B;
}

QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #0B1324;
    border: 1px solid #243554;
    border-radius: 10px;
    padding: 7px 10px;
    color: #F8FAFC;
    font-size: 13px;
    font-weight: 600;
    selection-background-color: #0284C7;
}

QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid #38BDF8;
    background-color: #0F1B33;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #0F172A;
    border: 1px solid #243554;
    color: #F8FAFC;
    selection-background-color: #0284C7;
    outline: none;
}

QPushButton#NavBtn {
    background-color: transparent;
    color: #94A3B8;
    border: none;
    border-radius: 9px;
    padding: 7px 16px;
    font-weight: 600;
    font-size: 13px;
}

QPushButton#NavBtn:hover {
    color: #F8FAFC;
    background-color: #16223B;
}

QPushButton#NavBtn[active="true"] {
    background-color: #0284C7;
    color: #FFFFFF;
}

QPushButton#PrimaryBtn {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #4F46E5);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#PrimaryBtn:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0EA5E9, stop:1 #6366F1);
}

QPushButton#PrimaryBtn:pressed {
    background-color: #0369A1;
}

QPushButton#SecondaryBtn {
    background-color: #182540;
    color: #E2E8F0;
    border: 1px solid #2A3F66;
    border-radius: 10px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton#SecondaryBtn:hover {
    background-color: #213256;
    border-color: #38BDF8;
    color: #FFFFFF;
}

QPushButton#PresetBtn {
    background-color: #0E1629;
    color: #CBD5E1;
    border: 1px solid #233554;
    border-radius: 9px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton#PresetBtn:hover {
    background-color: #192847;
    border-color: #38BDF8;
    color: #FFFFFF;
}

QProgressBar#ProbBar {
    background-color: #0B1324;
    border: 1px solid #1E2D4A;
    border-radius: 8px;
    height: 14px;
    text-align: center;
}

QProgressBar#ProbBar::chunk {
    border-radius: 7px;
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0EA5E9, stop:1 #6366F1);
}

QTextEdit {
    background-color: #0B1324;
    border: 1px solid #243554;
    border-radius: 10px;
    padding: 8px;
    color: #E2E8F0;
}

QScrollArea {
    border: none;
    background-color: transparent;
}
"""


# ---------------------------------------------------------------------------
# Background Worker Threads (keeps UI 100% responsive)
# ---------------------------------------------------------------------------
class CityForecastWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, city_name: str):
        super().__init__()
        self.city_name = city_name

    def run(self) -> None:
        try:
            data = fetch_city_forecast(self.city_name)
            self.finished.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))


class TurkeyMapWorker(QThread):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            data = fetch_turkey_map_weather()
            self.finished.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Custom Crisp Vector Weather Icon Widget (no missing external PNG files)
# ---------------------------------------------------------------------------
class WeatherBadgeWidget(QWidget):
    """Draws a crisp HiDPI weather illustration (idle / sunny / cloudy / rainy)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.state = "idle"  # idle | sunny | cloudy | rainy
        self.setFixedSize(96, 96)

    def set_state(self, state: str) -> None:
        self.state = state
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Soft circular glow backdrop
        bg_grad = QLinearGradient(0, 0, 96, 96)
        if self.state == "rainy":
            bg_grad.setColorAt(0.0, QColor(14, 165, 233, 45))
            bg_grad.setColorAt(1.0, QColor(99, 102, 241, 45))
            border_color = QColor(56, 189, 248, 110)
        elif self.state == "sunny":
            bg_grad.setColorAt(0.0, QColor(245, 158, 11, 45))
            bg_grad.setColorAt(1.0, QColor(16, 185, 129, 35))
            border_color = QColor(251, 191, 36, 115)
        elif self.state == "cloudy":
            bg_grad.setColorAt(0.0, QColor(245, 158, 11, 35))
            bg_grad.setColorAt(1.0, QColor(148, 163, 184, 40))
            border_color = QColor(245, 158, 11, 100)
        else:
            bg_grad.setColorAt(0.0, QColor(30, 41, 59, 130))
            bg_grad.setColorAt(1.0, QColor(15, 23, 42, 130))
            border_color = QColor(71, 85, 105, 120)

        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(border_color, 1.5))
        painter.drawEllipse(QRectF(4, 4, 88, 88))

        if self.state == "sunny":
            self._draw_sun(painter, 48, 48, 17, rays=True)
        elif self.state == "rainy":
            self._draw_cloud(painter, 48, 42, QColor("#E2E8F0"), QColor("#94A3B8"))
            self._draw_raindrops(painter)
        elif self.state == "cloudy":
            self._draw_sun(painter, 60, 34, 12, rays=False)
            self._draw_cloud(painter, 44, 52, QColor("#F1F5F9"), QColor("#CBD5E1"))
        else:
            self._draw_sun(painter, 58, 35, 11, rays=False)
            self._draw_cloud(painter, 45, 50, QColor("#CBD5E1"), QColor("#94A3B8"))

    def _draw_sun(self, painter: QPainter, cx: float, cy: float, r: float, rays: bool) -> None:
        if rays:
            painter.setPen(QPen(QColor("#FBBF24"), 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            import math

            for i in range(8):
                angle = math.radians(i * 45)
                x1 = cx + math.cos(angle) * (r + 5)
                y1 = cy + math.sin(angle) * (r + 5)
                x2 = cx + math.cos(angle) * (r + 11)
                y2 = cy + math.sin(angle) * (r + 11)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        sun_grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
        sun_grad.setColorAt(0.0, QColor("#FDE047"))
        sun_grad.setColorAt(1.0, QColor("#F59E0B"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(sun_grad))
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def _draw_cloud(self, painter: QPainter, cx: float, cy: float, top_c: QColor, bot_c: QColor) -> None:
        grad = QLinearGradient(cx, cy - 18, cx, cy + 16)
        grad.setColorAt(0.0, top_c)
        grad.setColorAt(1.0, bot_c)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))

        path = QPainterPath()
        path.addEllipse(QRectF(cx - 26, cy - 4, 24, 20))
        path.addEllipse(QRectF(cx - 14, cy - 16, 26, 26))
        path.addEllipse(QRectF(cx + 4, cy - 6, 22, 22))
        path.addRoundedRect(QRectF(cx - 18, cy + 2, 36, 14), 7, 7)
        painter.drawPath(path.simplified())

    def _draw_raindrops(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#38BDF8"), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        drops = [(34, 62, 30, 73), (48, 64, 44, 76), (62, 62, 58, 73)]
        for x1, y1, x2, y2 in drops:
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


# ---------------------------------------------------------------------------
# Optional Email Notification Dialog (uses .env credentials safely)
# ---------------------------------------------------------------------------
class EmailAlertDialog(QDialog):
    def __init__(self, subject: str, body: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Send Weather Forecast Email")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("✉️ Send Forecast Summary via SMTP")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASSWORD", "")
        default_to = os.getenv("ALERT_RECIPIENT_EMAIL", "")

        status_text = (
            f"Configured sender: {smtp_user}"
            if smtp_user and smtp_pass
            else "SMTP credentials not found. Copy .env.example to .env and set SMTP_USER & SMTP_PASSWORD."
        )
        status_lbl = QLabel(status_text)
        status_lbl.setWordWrap(True)
        status_lbl.setStyleSheet(
            "color: #10B981; font-size: 12px;"
            if smtp_user and smtp_pass
            else "color: #F59E0B; font-size: 12px;"
        )
        layout.addWidget(status_lbl)

        to_lbl = QLabel("Recipient Email")
        to_lbl.setObjectName("FieldLabel")
        layout.addWidget(to_lbl)

        self.to_input = QLineEdit(default_to)
        self.to_input.setPlaceholderText("recipient@example.com")
        layout.addWidget(self.to_input)

        subj_lbl = QLabel("Subject")
        subj_lbl.setObjectName("FieldLabel")
        layout.addWidget(subj_lbl)

        self.subject_input = QLineEdit(subject)
        layout.addWidget(self.subject_input)

        body_lbl = QLabel("Message Preview")
        body_lbl.setObjectName("FieldLabel")
        layout.addWidget(body_lbl)

        self.body_edit = QTextEdit()
        self.body_edit.setPlainText(body)
        self.body_edit.setFixedHeight(150)
        layout.addWidget(self.body_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        send_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        send_btn.setText("Send Email")
        send_btn.setObjectName("PrimaryBtn")
        send_btn.setEnabled(bool(smtp_user and smtp_pass))
        buttons.accepted.connect(self._send)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _send(self) -> None:
        recipient = self.to_input.text().strip()
        if not recipient or "@" not in recipient:
            QMessageBox.warning(self, "Invalid Email", "Please enter a valid recipient email address.")
            return

        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        try:
            smtp_port = int(os.getenv("SMTP_PORT") or "587")
        except ValueError:
            smtp_port = 587
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")

        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg["Subject"] = self.subject_input.text().strip()
        msg.attach(MIMEText(self.body_edit.toPlainText(), "plain", "utf-8"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            QMessageBox.information(self, "Email Sent", f"Forecast email sent to {recipient}.")
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Email Error", f"Could not send email:\n{exc}")


# ---------------------------------------------------------------------------
# Page 2: Interactive Real-Time Turkey Weather Map (81 Provinces)
# ---------------------------------------------------------------------------
class TurkeyMapPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.weather_data: list[dict[str, Any]] = []
        self.dark_tiles = True
        self.filter_mode = "all"
        self.has_loaded = False
        self.worker: TurkeyMapWorker | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 20)
        root.setSpacing(14)

        # Stats + Controls Bar
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self.stat_warmest = self._make_stat_pill("🔥 Warmest", "—")
        self.stat_coldest = self._make_stat_pill("❄️ Coldest", "—")
        self.stat_rainy = self._make_stat_pill("🌧️ Rainy Cities", "—")
        self.stat_updated = self._make_stat_pill("🕒 Status", "Not loaded")

        for pill in (self.stat_warmest, self.stat_coldest, self.stat_rainy, self.stat_updated):
            top_row.addWidget(pill)

        top_row.addStretch()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All 81 Cities", "Rainy Only", "Dry / Clear Only"])
        self.filter_combo.setFixedWidth(155)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        top_row.addWidget(self.filter_combo)

        self.theme_btn = QPushButton("☀️ Light Map")
        self.theme_btn.setObjectName("SecondaryBtn")
        self.theme_btn.clicked.connect(self._toggle_map_theme)
        top_row.addWidget(self.theme_btn)

        self.refresh_btn = QPushButton("↻ Refresh 81 Cities")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.refresh_btn.clicked.connect(self.load_map_data)
        top_row.addWidget(self.refresh_btn)

        root.addLayout(top_row)

        # Map Frame
        map_card = QFrame()
        map_card.setObjectName("Card")
        map_layout = QVBoxLayout(map_card)
        map_layout.setContentsMargins(6, 6, 6, 6)

        self.map_view = QWebEngineView()
        self.map_view.setStyleSheet("background-color: #0B1120; border-radius: 12px;")
        map_layout.addWidget(self.map_view)
        root.addWidget(map_card, stretch=1)

        self._render_placeholder("Click Refresh or switch to this tab to load live weather across 81 Turkish provinces.")

    def _make_stat_pill(self, label: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("StatPill")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(8)
        lbl = QLabel(label)
        lbl.setObjectName("MutedText")
        val = QLabel(value)
        val.setStyleSheet("font-weight: 700; color: #F8FAFC; font-size: 12px;")
        val.setObjectName("pill_value")
        lay.addWidget(lbl)
        lay.addWidget(val)
        return frame

    def _set_pill_value(self, pill: QFrame, text: str) -> None:
        val = pill.findChild(QLabel, "pill_value")
        if val:
            val.setText(text)

    def ensure_loaded(self) -> None:
        if not self.has_loaded and self.worker is None:
            self.load_map_data()

    def load_map_data(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Loading...")
        self._set_pill_value(self.stat_updated, "Fetching 81 cities...")
        self._render_placeholder("Fetching live weather data for all 81 Turkish provinces...")

        self.worker = TurkeyMapWorker()
        self.worker.finished.connect(self._on_map_loaded)
        self.worker.failed.connect(self._on_map_failed)
        self.worker.start()

    def _on_map_loaded(self, data: list[dict[str, Any]]) -> None:
        self.weather_data = data
        self.has_loaded = True
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("↻ Refresh 81 Cities")

        if data:
            warmest = max(data, key=lambda c: c["temp"])
            coldest = min(data, key=lambda c: c["temp"])
            rainy_count = sum(1 for c in data if c["is_rainy"])
            self._set_pill_value(self.stat_warmest, f"{warmest['city']} ({warmest['temp']}°C)")
            self._set_pill_value(self.stat_coldest, f"{coldest['city']} ({coldest['temp']}°C)")
            self._set_pill_value(self.stat_rainy, f"{rainy_count} / {len(data)}")
            self._set_pill_value(self.stat_updated, datetime.datetime.now().strftime("%H:%M"))

        self._render_map()

    def _on_map_failed(self, err: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("↻ Refresh 81 Cities")
        self._set_pill_value(self.stat_updated, "Offline / Error")
        self._render_placeholder(
            f"Could not fetch live map data. Please check your internet connection.<br><small>{err}</small>"
        )

    def _on_filter_changed(self, idx: int) -> None:
        self.filter_mode = ["all", "rainy", "dry"][idx]
        if self.weather_data:
            self._render_map()

    def _toggle_map_theme(self) -> None:
        self.dark_tiles = not self.dark_tiles
        self.theme_btn.setText("☀️ Light Map" if self.dark_tiles else "🌙 Dark Map")
        if self.weather_data:
            self._render_map()

    def _render_placeholder(self, message: str) -> None:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8" /></head>
        <body style="margin:0; height:100vh; display:flex; align-items:center; justify-content:center;
                     background:#0B1120; color:#94A3B8; font-family:'Segoe UI',sans-serif; text-align:center;">
            <div style="padding:24px 32px; background:#131C31; border:1px solid #1E2D4A; border-radius:16px; max-width:480px;">
                <div style="font-size:32px; margin-bottom:10px;">🗺️</div>
                <div style="font-size:14px; line-height:1.6;">{message}</div>
            </div>
        </body>
        </html>
        """
        self.map_view.setHtml(html)

    def _render_map(self) -> None:
        if self.filter_mode == "rainy":
            filtered = [c for c in self.weather_data if c["is_rainy"]]
        elif self.filter_mode == "dry":
            filtered = [c for c in self.weather_data if not c["is_rainy"]]
        else:
            filtered = self.weather_data

        tile_layer_js = (
            """
                L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
                    maxZoom: 16,
                    attribution: '&copy; Esri &copy; OpenStreetMap | Weather: Open-Meteo'
                }).addTo(map);
                L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
                    maxZoom: 16
                }).addTo(map);
            """
            if self.dark_tiles
            else """
                L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
                    maxZoom: 18,
                    attribution: '&copy; Esri &copy; OpenStreetMap | Weather: Open-Meteo'
                }).addTo(map);
            """
        )
        bg_color = "#1B1E24" if self.dark_tiles else "#E2E8F0"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                html, body, #map {{
                    height: 100%;
                    width: 100%;
                    margin: 0;
                    padding: 0;
                    background: {bg_color};
                    font-family: 'Segoe UI', sans-serif;
                }}
                .temp-pill {{
                    display: inline-flex;
                    align-items: center;
                    gap: 3px;
                    padding: 3px 7px;
                    border-radius: 999px;
                    font-size: 11px;
                    font-weight: 700;
                    color: #ffffff;
                    border: 1.5px solid rgba(255,255,255,0.4);
                    box-shadow: 0 3px 10px rgba(0,0,0,0.45);
                    white-space: nowrap;
                    transform: translate(-50%, -50%);
                }}
                .leaflet-popup-content-wrapper {{
                    background: #0F172A;
                    color: #F8FAFC;
                    border: 1px solid #263A5E;
                    border-radius: 12px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                }}
                .leaflet-popup-tip {{
                    background: #0F172A;
                }}
                .popup-title {{
                    font-size: 15px;
                    font-weight: 700;
                    margin-bottom: 4px;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }}
                .popup-desc {{
                    font-size: 12px;
                    color: #94A3B8;
                    margin-bottom: 8px;
                }}
                .popup-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 6px 14px;
                    font-size: 12px;
                }}
                .popup-grid span {{
                    color: #94A3B8;
                }}
                .popup-grid b {{
                    color: #F8FAFC;
                }}
                .legend {{
                    background: rgba(15, 23, 42, 0.92);
                    border: 1px solid #263A5E;
                    border-radius: 10px;
                    padding: 8px 12px;
                    color: #E2E8F0;
                    font-size: 11px;
                    line-height: 1.7;
                }}
                .dot {{
                    display: inline-block;
                    width: 9px;
                    height: 9px;
                    border-radius: 50%;
                    margin-right: 5px;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map', {{ zoomControl: true }}).setView([39.05, 35.35], 6);

                {tile_layer_js}

                function tempColor(t, isRainy) {{
                    if (isRainy) return 'linear-gradient(135deg, #0284C7, #4F46E5)';
                    if (t <= 2) return '#2563EB';
                    if (t <= 10) return '#0891B2';
                    if (t <= 18) return '#059669';
                    if (t <= 26) return '#D97706';
                    return '#DC2626';
                }}

                var cities = {json.dumps(filtered, ensure_ascii=False)};

                cities.forEach(function(c) {{
                    var badge = (c.is_rainy ? '💧 ' : '') + Math.round(c.temp) + '°';
                    var icon = L.divIcon({{
                        className: '',
                        html: '<div class="temp-pill" style="background:' + tempColor(c.temp, c.is_rainy) + ';">' + badge + '</div>',
                        iconSize: [0, 0]
                    }});

                    var popupHtml =
                        '<div class="popup-title">' + c.emoji + ' ' + c.city + '</div>' +
                        '<div class="popup-desc">' + c.desc + '</div>' +
                        '<div class="popup-grid">' +
                            '<div><span>Temp:</span> <b>' + c.temp + ' °C</b></div>' +
                            '<div><span>Feels:</span> <b>' + c.feels_like + ' °C</b></div>' +
                            '<div><span>Humidity:</span> <b>' + c.humidity + '%</b></div>' +
                            '<div><span>Wind:</span> <b>' + c.wind + ' m/s</b></div>' +
                            '<div><span>Clouds:</span> <b>' + c.cloud + '%</b></div>' +
                            '<div><span>Rain:</span> <b>' + c.precip + ' mm</b></div>' +
                        '</div>';

                    L.marker([c.lat, c.lon], {{ icon: icon }})
                        .bindTooltip(c.city + ': ' + c.temp + '°C · ' + c.desc, {{ direction: 'top', offset: [0, -14] }})
                        .bindPopup(popupHtml)
                        .addTo(map);
                }});

                var legend = L.control({{ position: 'bottomright' }});
                legend.onAdd = function() {{
                    var div = L.DomUtil.create('div', 'legend');
                    div.innerHTML =
                        '<b>Legend</b><br>' +
                        '<span class="dot" style="background:#0284C7;"></span>Rain / Precip (💧)<br>' +
                        '<span class="dot" style="background:#2563EB;"></span>≤ 2°C &nbsp;' +
                        '<span class="dot" style="background:#0891B2;"></span>3–10°C<br>' +
                        '<span class="dot" style="background:#059669;"></span>11–18°C &nbsp;' +
                        '<span class="dot" style="background:#D97706;"></span>19–26°C &nbsp;' +
                        '<span class="dot" style="background:#DC2626;"></span>> 26°C';
                    return div;
                }};
                legend.addTo(map);
            </script>
        </body>
        </html>
        """
        self.map_view.setHtml(html)


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------
class WeatherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rain Prediction — Machine Learning Forecast")
        self.resize(1220, 750)
        self.setMinimumSize(1040, 660)
        self.setStyleSheet(STYLESHEET)

        self.model: Any = None
        self.model_name: str = "Ensemble Classifier"
        self.feature_names: list[str] = DEFAULT_FEATURES
        self.model_metrics: dict[str, Any] = {}
        self.threshold: float = 0.50
        self.latest_forecast: dict[str, Any] | None = None
        self.forecast_worker: CityForecastWorker | None = None
        self._retired_workers: list[QThread] = []
        self._expected_city: str = ""

        self._load_model()
        self._build_ui()

        # Run an immediate general prediction on default atmospheric values (no city required!)
        self.predict_manual()

    # ---- Model Loading ----
    def _load_model(self) -> None:
        model_path = os.path.join(PROJECT_ROOT, "models", "rain_model.pkl")
        if os.path.exists(model_path):
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded["model"]
                self.model_name = loaded.get("model_name", "Ensemble Classifier")
                self.feature_names = loaded.get("feature_names", DEFAULT_FEATURES)
                self.model_metrics = loaded.get("metrics", {})
                self.threshold = float(loaded.get("threshold", 0.50))
            else:
                self.model = loaded
                if hasattr(loaded, "feature_names_in_"):
                    self.feature_names = list(loaded.feature_names_in_)

    # ---- UI Construction ----
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_top_bar())

        self.stack = QStackedWidget()
        self.dashboard_page = self._build_dashboard_page()
        self.map_page = TurkeyMapPage()

        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.map_page)
        outer.addWidget(self.stack, stretch=1)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(66)

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 8, 20, 8)
        lay.setSpacing(16)

        # Brand
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        title = QLabel("🌧️  Rain Prediction ML")
        title.setObjectName("AppTitle")
        sub = QLabel("General Atmospheric Rain Probability Model · 5-Model Ensemble")
        sub.setObjectName("AppSubtitle")
        brand_box.addWidget(title)
        brand_box.addWidget(sub)
        lay.addLayout(brand_box)

        lay.addStretch()

        # Segmented Navigation
        nav_frame = QFrame()
        nav_frame.setObjectName("NavContainer")
        nav_lay = QHBoxLayout(nav_frame)
        nav_lay.setContentsMargins(4, 4, 4, 4)
        nav_lay.setSpacing(4)

        self.nav_pred_btn = QPushButton("⚡  Rain Predictor")
        self.nav_pred_btn.setObjectName("NavBtn")
        self.nav_pred_btn.setProperty("active", True)
        self.nav_pred_btn.clicked.connect(lambda: self._switch_page(0))

        self.nav_map_btn = QPushButton("🗺️  Live Weather Map")
        self.nav_map_btn.setObjectName("NavBtn")
        self.nav_map_btn.setProperty("active", False)
        self.nav_map_btn.clicked.connect(lambda: self._switch_page(1))

        nav_lay.addWidget(self.nav_pred_btn)
        nav_lay.addWidget(self.nav_map_btn)
        lay.addWidget(nav_frame)

        return bar

    def _build_dashboard_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        main_lay = QHBoxLayout(container)
        main_lay.setContentsMargins(20, 18, 20, 20)
        main_lay.setSpacing(18)

        # ---- LEFT COLUMN: Atmospheric Inputs ----
        input_card = QFrame()
        input_card.setObjectName("Card")
        input_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_lay = QVBoxLayout(input_card)
        left_lay.setContentsMargins(22, 20, 22, 20)
        left_lay.setSpacing(14)

        header_row = QHBoxLayout()
        hdr_title = QLabel("Atmospheric Parameters")
        hdr_title.setObjectName("SectionTitle")
        header_row.addWidget(hdr_title)
        header_row.addStretch()

        self.month_combo = QComboBox()
        self.month_combo.addItems(MONTH_NAMES)
        tomorrow_month = (datetime.date.today() + datetime.timedelta(days=1)).month
        self.month_combo.setCurrentIndex(tomorrow_month - 1)
        self.month_combo.setFixedWidth(130)
        header_row.addWidget(self.month_combo)
        left_lay.addLayout(header_row)

        # Quick Scenario Presets
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)

        presets = [
            ("☀️ Clear Sky", (27.5, 16.0, 2.5, 10.0, 42.0, 1021.0, 7.2)),
            ("⛅ Partly Cloudy", (19.5, 12.0, 3.8, 58.0, 68.0, 1013.5, 4.2)),
            ("🌧️ Storm Front", (12.0, 7.5, 7.6, 94.0, 88.0, 1001.5, 1.4)),
        ]
        for label, vals in presets:
            btn = QPushButton(label)
            btn.setObjectName("PresetBtn")
            btn.clicked.connect(lambda checked=False, v=vals: self._apply_preset(v))
            preset_row.addWidget(btn)
        left_lay.addLayout(preset_row)

        # 2-Column SpinBox Grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        field_specs = [
            ("Max_Temperature", "Max Temperature", -30.0, 55.0, 20.0, 0.5, " °C"),
            ("Min_Temperature", "Min Temperature", -40.0, 45.0, 12.0, 0.5, " °C"),
            ("WindSpeed", "Wind Speed", 0.0, 50.0, 3.5, 0.2, " m/s"),
            ("CloudCover", "Cloud Cover", 0.0, 100.0, 45.0, 2.0, " %"),
            ("Humidity", "Relative Humidity", 0.0, 100.0, 65.0, 2.0, " %"),
            ("Pressure", "Sea-Level Pressure", 940.0, 1060.0, 1013.0, 0.5, " hPa"),
            ("Uvi", "Max UV Index", 0.0, 15.0, 4.5, 0.2, ""),
        ]

        self.spinboxes: dict[str, QDoubleSpinBox] = {}
        for idx, (key, title, lo, hi, default, step, suffix) in enumerate(field_specs):
            box = QVBoxLayout()
            box.setSpacing(5)
            lbl = QLabel(title)
            lbl.setObjectName("FieldLabel")
            spin = QDoubleSpinBox()
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setSingleStep(step)
            spin.setDecimals(1)
            if suffix:
                spin.setSuffix(suffix)
            spin.setMinimumHeight(38)
            box.addWidget(lbl)
            box.addWidget(spin)
            self.spinboxes[key] = spin

            row = idx // 2
            col = idx % 2
            if idx == len(field_specs) - 1:
                grid.addLayout(box, row, 0, 1, 2)
            else:
                grid.addLayout(box, row, col)

        left_lay.addLayout(grid)
        left_lay.addStretch()

        # Predict CTA
        self.predict_btn = QPushButton("⚡  Predict Rain Probability")
        self.predict_btn.setObjectName("PrimaryBtn")
        self.predict_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.predict_btn.clicked.connect(self.predict_manual)
        left_lay.addWidget(self.predict_btn)

        # Model Metadata Footer
        if self.model_metrics:
            acc = self.model_metrics.get("accuracy", 0) * 100
            auc = self.model_metrics.get("roc_auc", 0)
            total = self.model_metrics.get("total_samples", 0)
            meta_txt = (
                f"Ensemble (5 models) · Acc {acc:.1f}% · ROC-AUC {auc:.2f} · {total:,} records"
            )
        elif self.model is not None:
            meta_txt = f"Model: {self.model_name} loaded"
        else:
            meta_txt = "⚠️ Model not found — run `python -m src.train_model`"

        self.model_meta_lbl = QLabel(meta_txt)
        self.model_meta_lbl.setObjectName("MutedText")
        self.model_meta_lbl.setWordWrap(True)
        self.model_meta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_lay.addWidget(self.model_meta_lbl)

        # ---- RIGHT COLUMN: Prediction Hero + Live Forecast ----
        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        # Hero Prediction Card
        hero_card = QFrame()
        hero_card.setObjectName("HeroCard")
        hero_lay = QVBoxLayout(hero_card)
        hero_lay.setContentsMargins(22, 20, 22, 20)
        hero_lay.setSpacing(14)

        hero_top = QHBoxLayout()
        hero_top.setSpacing(18)

        self.weather_badge = WeatherBadgeWidget()
        hero_top.addWidget(self.weather_badge)

        verdict_col = QVBoxLayout()
        verdict_col.setSpacing(4)

        badge_row = QHBoxLayout()
        self.risk_badge = QLabel("AWAITING INPUT")
        self.risk_badge.setObjectName("RiskBadge")
        badge_row.addWidget(self.risk_badge)
        badge_row.addStretch()
        verdict_col.addLayout(badge_row)

        self.verdict_label = QLabel("Ready to Predict")
        self.verdict_label.setObjectName("VerdictTitle")
        verdict_col.addWidget(self.verdict_label)

        self.insight_label = QLabel("Adjust parameters or sync tomorrow's live forecast to calculate rain probability.")
        self.insight_label.setWordWrap(True)
        self.insight_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        verdict_col.addWidget(self.insight_label)
        hero_top.addLayout(verdict_col, stretch=1)

        self.prob_big_label = QLabel("—%")
        self.prob_big_label.setObjectName("ProbabilityBig")
        self.prob_big_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hero_top.addWidget(self.prob_big_label)

        hero_lay.addLayout(hero_top)

        self.prob_bar = QProgressBar()
        self.prob_bar.setObjectName("ProbBar")
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setValue(0)
        self.prob_bar.setTextVisible(False)
        hero_lay.addWidget(self.prob_bar)

        right_col.addWidget(hero_card)

        # Atmospheric Physics & Optional Global Auto-Fill Card
        forecast_card = QFrame()
        forecast_card.setObjectName("Card")
        fc_lay = QVBoxLayout(forecast_card)
        fc_lay.setContentsMargins(22, 18, 22, 18)
        fc_lay.setSpacing(14)

        fc_header = QHBoxLayout()
        fc_title_col = QVBoxLayout()
        fc_title_col.setSpacing(2)
        self.fc_city_title = QLabel("Derived Atmospheric Indicators")
        self.fc_city_title.setObjectName("SectionTitle")
        self.fc_current_sub = QLabel("Computed in real time from your parameters (any location)")
        self.fc_current_sub.setObjectName("MutedText")
        self.fc_current_sub.setWordWrap(True)
        fc_title_col.addWidget(self.fc_city_title)
        fc_title_col.addWidget(self.fc_current_sub)
        fc_header.addLayout(fc_title_col, stretch=1)

        self.email_btn = QPushButton("✉️ Email")
        self.email_btn.setObjectName("SecondaryBtn")
        self.email_btn.clicked.connect(self._open_email_dialog)
        fc_header.addWidget(self.email_btn)
        fc_lay.addLayout(fc_header)

        # 6 Derived Physics Metric Tiles (3x2) — live for ANY manual inputs
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        self.metric_labels: dict[str, QLabel] = {}
        tile_defs = [
            ("temp_stats", "🌡️ Mean Temp · ΔT"),
            ("dew_point", "💧 Dew Point"),
            ("dew_spread", "🌫️ Dew Spread"),
            ("cloud_hum", "☁️ Cloud × Humidity"),
            ("pressure_dev", "🧭 Pressure Anomaly"),
            ("storm_index", "⛈️ Storm Index"),
        ]
        for idx, (key, label) in enumerate(tile_defs):
            tile = QFrame()
            tile.setObjectName("MetricTile")
            t_lay = QVBoxLayout(tile)
            t_lay.setContentsMargins(12, 10, 12, 10)
            t_lay.setSpacing(3)
            t_lbl = QLabel(label)
            t_lbl.setObjectName("MetricTitle")
            t_val = QLabel("—")
            t_val.setObjectName("MetricValue")
            t_lay.addWidget(t_lbl)
            t_lay.addWidget(t_val)
            self.metric_labels[key] = t_val
            metrics_grid.addWidget(tile, idx // 3, idx % 3)

        fc_lay.addLayout(metrics_grid)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #1E2D4A; max-height: 1px; border: none; margin: 4px 0;")
        fc_lay.addWidget(divider)

        # Optional Global City Auto-Fill Bar
        opt_lbl = QLabel("Optional: Auto-Fill Parameters from Live Forecast (Any City Worldwide)")
        opt_lbl.setObjectName("FieldLabel")
        fc_lay.addWidget(opt_lbl)

        lookup_row = QHBoxLayout()
        lookup_row.setSpacing(10)
        self.city_input = QLineEdit()
        self.city_input.setPlaceholderText("e.g. London, Tokyo, New York, Berlin, Istanbul...")
        self.city_input.returnPressed.connect(lambda: self.fetch_selected_city_forecast(auto_fill=True))
        lookup_row.addWidget(self.city_input, stretch=1)

        self.sync_btn = QPushButton("🌍 Auto-Fill")
        self.sync_btn.setObjectName("SecondaryBtn")
        self.sync_btn.clicked.connect(lambda: self.fetch_selected_city_forecast(auto_fill=True))
        lookup_row.addWidget(self.sync_btn)
        fc_lay.addLayout(lookup_row)

        self.live_status_lbl = QLabel("Direct mode active — enter any atmospheric values on the left and click Predict.")
        self.live_status_lbl.setObjectName("MutedText")
        self.live_status_lbl.setWordWrap(True)
        fc_lay.addWidget(self.live_status_lbl)

        right_col.addWidget(forecast_card, stretch=1)

        main_lay.addWidget(input_card, stretch=45)
        main_lay.addLayout(right_col, stretch=55)

        scroll.setWidget(container)
        return scroll

    # ---- Navigation ----
    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.nav_pred_btn.setProperty("active", index == 0)
        self.nav_map_btn.setProperty("active", index == 1)
        for btn in (self.nav_pred_btn, self.nav_map_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if index == 1:
            self.map_page.ensure_loaded()

    # ---- Presets & Prediction ----
    def _apply_preset(self, values: tuple[float, ...]) -> None:
        keys = [
            "Max_Temperature",
            "Min_Temperature",
            "WindSpeed",
            "CloudCover",
            "Humidity",
            "Pressure",
            "Uvi",
        ]
        for key, val in zip(keys, values):
            self.spinboxes[key].setValue(val)
        self.predict_manual()

    def predict_manual(self) -> None:
        if self.model is None:
            self.verdict_label.setText("Model Not Loaded")
            self.insight_label.setText("Run `python -m src.train_model` to generate models/rain_model.pkl.")
            return

        max_t = self.spinboxes["Max_Temperature"].value()
        min_t = self.spinboxes["Min_Temperature"].value()
        if min_t > max_t:
            min_t, max_t = max_t, min_t
            self.spinboxes["Max_Temperature"].setValue(max_t)
            self.spinboxes["Min_Temperature"].setValue(min_t)

        wind = self.spinboxes["WindSpeed"].value()
        cloud = self.spinboxes["CloudCover"].value()
        humidity = self.spinboxes["Humidity"].value()
        pressure = self.spinboxes["Pressure"].value()
        uvi = self.spinboxes["Uvi"].value()
        temp_diff = round(max_t - min_t, 2)
        month = self.month_combo.currentIndex() + 1

        engineered_df = build_single_feature_row(
            max_temp=max_t,
            min_temp=min_t,
            wind_speed=wind,
            cloud_cover=cloud,
            humidity=humidity,
            pressure=pressure,
            uv_index=uvi,
            month=month,
        )
        # Support both engineered 18-feature models and legacy 9-feature models
        if "Month" in self.feature_names and "Month" not in engineered_df.columns:
            engineered_df["Month"] = month
        row_df = engineered_df[self.feature_names]

        prob_raw = float(self.model.predict_proba(row_df)[0][1])
        pred = 1 if prob_raw >= self.threshold else 0
        prob = prob_raw * 100.0

        self.prob_big_label.setText(f"{prob:.1f}%")
        self.prob_bar.setValue(int(round(prob)))

        if pred == 1:
            self.weather_badge.set_state("rainy")
            self.verdict_label.setText("Rain Expected")
            self.risk_badge.setText("HIGH RAIN RISK" if prob >= 65 else "MODERATE-HIGH RISK")
            self.risk_badge.setStyleSheet(
                "font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px;"
                "background-color: rgba(14,165,233,0.22); color: #38BDF8;"
            )
            self.prob_big_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #38BDF8;")
            self.prob_bar.setStyleSheet(
                "QProgressBar#ProbBar::chunk { border-radius: 7px; "
                "background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0284C7, stop:1 #6366F1); }"
            )
        elif prob >= 30.0:
            self.weather_badge.set_state("cloudy")
            self.verdict_label.setText("Possible Showers")
            self.risk_badge.setText("MODERATE RISK")
            self.risk_badge.setStyleSheet(
                "font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px;"
                "background-color: rgba(245,158,11,0.22); color: #FBBF24;"
            )
            self.prob_big_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #FBBF24;")
            self.prob_bar.setStyleSheet(
                "QProgressBar#ProbBar::chunk { border-radius: 7px; "
                "background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #D97706, stop:1 #FBBF24); }"
            )
        else:
            self.weather_badge.set_state("sunny")
            self.verdict_label.setText("No Rain Expected")
            self.risk_badge.setText("LOW RAIN RISK")
            self.risk_badge.setStyleSheet(
                "font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px;"
                "background-color: rgba(16,185,129,0.22); color: #34D399;"
            )
            self.prob_big_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #34D399;")
            self.prob_bar.setStyleSheet(
                "QProgressBar#ProbBar::chunk { border-radius: 7px; "
                "background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #059669, stop:1 #34D399); }"
            )

        # Update real-time derived atmospheric physics tiles
        mean_t = float(engineered_df["Temp_Mean"].iloc[0])
        dew_pt = float(engineered_df["Dew_Point"].iloc[0])
        dew_sp = float(engineered_df["Dew_Spread"].iloc[0])
        cloud_hum = float(engineered_df["Cloud_Humidity"].iloc[0])
        press_dev = float(engineered_df["Pressure_Dev"].iloc[0])
        storm_idx = float(engineered_df["Storm_Index"].iloc[0])

        self.metric_labels["temp_stats"].setText(f"{mean_t:.1f}°C · Δ{temp_diff:.1f}°C")
        self.metric_labels["dew_point"].setText(f"{dew_pt:.1f} °C")
        self.metric_labels["dew_spread"].setText(f"{dew_sp:.1f} °C")
        self.metric_labels["cloud_hum"].setText(f"{cloud_hum:.1f}%")
        self.metric_labels["pressure_dev"].setText(f"{press_dev:+.1f} hPa")
        self.metric_labels["storm_index"].setText(f"{storm_idx:.1f}")

        # Meteorological explanation
        drivers = []
        if cloud >= 70:
            drivers.append(f"dense cloud cover ({cloud:.0f}%)")
        elif cloud <= 25:
            drivers.append(f"clear skies ({cloud:.0f}% cloud)")
        if humidity >= 75:
            drivers.append(f"elevated humidity ({humidity:.0f}%)")
        elif humidity <= 50:
            drivers.append(f"dry air ({humidity:.0f}% humidity)")
        if pressure <= 1007:
            drivers.append(f"low pressure ({pressure:.1f} hPa)")
        elif pressure >= 1020:
            drivers.append(f"high pressure ({pressure:.1f} hPa)")
        if wind >= 6.0:
            drivers.append(f"strong winds ({wind:.1f} m/s)")

        if drivers:
            self.insight_label.setText("Key drivers: " + ", ".join(drivers).capitalize() + ".")
        else:
            self.insight_label.setText(
                f"Balanced atmospheric profile (ΔT {temp_diff:.1f}°C, {MONTH_NAMES[month - 1]})."
            )

    # ---- Optional Live Forecast Auto-Fill (Any City Worldwide) ----
    def fetch_selected_city_forecast(
        self, auto_fill: bool = False, city_override: str | None = None
    ) -> None:
        query = (city_override if city_override is not None else self.city_input.text()).strip()
        if not query:
            self.live_status_lbl.setText("Enter any city name (e.g. London, Tokyo, Istanbul) to auto-fill live values.")
            return

        self._expected_city = query

        if self.forecast_worker is not None and self.forecast_worker.isRunning():
            prev = self.forecast_worker
            self._retired_workers.append(prev)
            prev.finished.connect(lambda _d=None, w=prev: self._cleanup_retired_worker(w))
            prev.failed.connect(lambda _e=None, w=prev: self._cleanup_retired_worker(w))

        self.sync_btn.setEnabled(False)
        self.sync_btn.setText("Loading...")
        self.live_status_lbl.setText(f"Fetching tomorrow's live parameters for '{query}'...")

        worker = CityForecastWorker(query)
        worker.finished.connect(
            lambda data, fill=auto_fill, req_city=query: self._handle_forecast_result(data, fill, req_city)
        )
        worker.failed.connect(
            lambda err, req_city=query: self._handle_forecast_error(err, req_city)
        )
        self.forecast_worker = worker
        worker.start()

    def _cleanup_retired_worker(self, worker: QThread) -> None:
        if worker in self._retired_workers:
            self._retired_workers.remove(worker)

    def _handle_forecast_result(self, data: dict[str, Any], auto_fill: bool, req_city: str) -> None:
        if req_city != self._expected_city:
            return
        self._on_forecast_loaded(data, auto_fill)

    def _handle_forecast_error(self, err: str, req_city: str) -> None:
        if req_city != self._expected_city:
            return
        self._on_forecast_failed(err)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Gracefully stop background threads on window close."""
        workers: list[QThread | None] = [
            self.forecast_worker,
            self.map_page.worker,
            *self._retired_workers,
        ]
        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                if not worker.wait(1500):
                    worker.terminate()
                    worker.wait(500)
        super().closeEvent(event)

    def _on_forecast_loaded(self, data: dict[str, Any], auto_fill: bool) -> None:
        self.latest_forecast = data
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("🌍 Auto-Fill")

        curr = data["current"]
        tmrw = data["tomorrow"]
        self.live_status_lbl.setText(
            f"Filled from {data['city']} ({tmrw['date']}): {tmrw['emoji']} {tmrw['condition']} · "
            f"Now {curr['temp']}°C · Expected precip {tmrw['precip_mm']} mm ({tmrw['precip_prob']}%)"
        )

        if auto_fill:
            self.spinboxes["Max_Temperature"].setValue(tmrw["max_temp"])
            self.spinboxes["Min_Temperature"].setValue(tmrw["min_temp"])
            self.spinboxes["WindSpeed"].setValue(tmrw["wind_speed"])
            self.spinboxes["CloudCover"].setValue(tmrw["cloud_cover"])
            self.spinboxes["Humidity"].setValue(tmrw["humidity"])
            self.spinboxes["Pressure"].setValue(tmrw["pressure"])
            self.spinboxes["Uvi"].setValue(tmrw["uv_index"])
            self.month_combo.setCurrentIndex(tmrw["month"] - 1)
            self.predict_manual()

    def _on_forecast_failed(self, err: str) -> None:
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("🌍 Auto-Fill")
        self.live_status_lbl.setText(f"Could not fetch live data ({err[:70]}). Direct prediction still works.")
        self.predict_manual()

    # ---- Email Alert Dialog ----
    def _open_email_dialog(self) -> None:
        max_t = self.spinboxes["Max_Temperature"].value()
        min_t = self.spinboxes["Min_Temperature"].value()
        wind = self.spinboxes["WindSpeed"].value()
        cloud = self.spinboxes["CloudCover"].value()
        humidity = self.spinboxes["Humidity"].value()
        pressure = self.spinboxes["Pressure"].value()
        uvi = self.spinboxes["Uvi"].value()
        month = self.month_combo.currentText()

        location_line = (
            f"Reference City : {self.latest_forecast['city']} ({self.latest_forecast['tomorrow']['date']})\n"
            if self.latest_forecast
            else "Mode           : General Direct Atmospheric Prediction\n"
        )
        subject = f"Rain Prediction Summary — {self.verdict_label.text()} ({self.prob_big_label.text()})"
        body = (
            f"Rain Prediction ML Summary\n"
            f"{'=' * 44}\n"
            f"{location_line}"
            f"Prediction     : {self.verdict_label.text()} ({self.prob_big_label.text()})\n"
            f"Temperature    : {min_t:.1f} °C – {max_t:.1f} °C\n"
            f"Wind Speed     : {wind:.1f} m/s\n"
            f"Cloud Cover    : {cloud:.0f}%\n"
            f"Humidity       : {humidity:.0f}%\n"
            f"Pressure (MSL) : {pressure:.1f} hPa\n"
            f"Max UV Index   : {uvi:.1f} ({month})\n\n"
            f"{self.insight_label.text()}\n"
        )

        dialog = EmailAlertDialog(subject, body, parent=self)
        dialog.exec()


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = WeatherApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
