"""
visualize_2d.py
---------------
Vẽ kết quả tracking lên sơ đồ sân 2D (tactical map) và ghép với khung hình camera.

  tracks_to_pitch   : all_tracks (bbox pixel) → {frame: {tid: (x_m, y_m)}} qua homography
  PitchCanvas       : vẽ sân 105 × 68 m, chấm cầu thủ, vệt di chuyển, HUD
  draw_tracks       : vẽ bbox + ID lên khung hình camera
  side_by_side      : ghép camera (trên) + sơ đồ sân (dưới) thành một khung hình
  open_writer       : VideoWriter .mp4 (mp4v)

Phần vẽ sân được tách từ Visualize.ipynb; tọa độ sân dùng chung box_to_pitch_xy
với bước đánh giá (pitch_eval.py) nên hình minh họa và số liệu nhất quán.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pitch_localization import box_to_pitch_xy

PITCH_W_M = 105.0          # chiều dài sân (m), trục X
PITCH_H_M = 68.0           # chiều rộng sân (m), trục Y
OUT_MARGIN_M = 5.0         # điểm chiếu ra ngoài sân quá 5 m thì bỏ
MAX_STEP_M = 3.0           # bước nhảy > 3 m giữa 2 frame liên tiếp = đổi ID, ngắt vệt

COLOR_GRASS_DARK  = (34, 85, 34)
COLOR_GRASS_LIGHT = (45, 100, 45)
COLOR_LINE        = (220, 220, 220)
COLOR_HUD_BG      = (10, 20, 10)

_PALETTE = [
    (255, 80, 80), (255, 165, 0), (255, 215, 0), (120, 230, 80),
    (50, 220, 120), (0, 200, 180), (30, 160, 255), (80, 100, 255),
    (160, 80, 255), (255, 60, 180), (255, 140, 100), (80, 200, 200),
    (200, 100, 255), (255, 100, 140), (100, 200, 100), (255, 200, 50),
    (50, 180, 255), (255, 50, 100), (180, 255, 80), (100, 100, 255),
    (255, 180, 50), (80, 255, 160), (200, 80, 200), (255, 120, 50),
]


def color_of(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ── Bbox → tọa độ sân ─────────────────────────────────────────────────────────

def tracks_to_pitch(all_tracks: Dict, H: np.ndarray) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    all_tracks : {tid: {'boxes': [tlbr | None theo frame], 'frames': [...]}}
    Trả về     : {frame_idx: {tid: (x_m, y_m)}} — điểm chân (bottom-center) chiếu qua H,
                 bỏ điểm nằm ngoài sân quá OUT_MARGIN_M.
    """
    per_frame: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for tid, data in all_tracks.items():
        boxes = data["boxes"]
        for f in data["frames"]:
            box = boxes[f] if f < len(boxes) else None
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x, y = box_to_pitch_xy([x1, y1, x2 - x1, y2 - y1], H=H)
            if -OUT_MARGIN_M <= x <= PITCH_W_M + OUT_MARGIN_M and -OUT_MARGIN_M <= y <= PITCH_H_M + OUT_MARGIN_M:
                per_frame.setdefault(f, {})[tid] = (float(x), float(y))
    return per_frame


# ── Sơ đồ sân ─────────────────────────────────────────────────────────────────

class PitchCanvas:
    """Sơ đồ sân 2D. Gọi render() mỗi frame theo thứ tự thời gian (giữ vệt di chuyển)."""

    def __init__(self, width: int = 1280, hud_h: int = 56, margin: int = 40, trail_len: int = 40):
        self.margin = margin
        self.hud_h = hud_h
        self.scale = (width - 2 * margin) / PITCH_W_M            # px / m (giữ đúng tỉ lệ sân)
        self.width = width
        self.height = int(hud_h + 2 * margin + PITCH_H_M * self.scale)
        self.trail_len = trail_len
        self.template = self._draw_pitch()
        self.trails: Dict[int, deque] = {}

    def to_px(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return (int(round(self.margin + x_m * self.scale)),
                int(round(self.hud_h + self.margin + y_m * self.scale)))

    def _draw_pitch(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = COLOR_GRASS_DARK
        p, s, lc = self.to_px, self.scale, COLOR_LINE

        n = 14
        for i in range(n):                                       # dải cỏ đậm/nhạt
            x0, _ = p(i * PITCH_W_M / n, 0)
            x1, _ = p((i + 1) * PITCH_W_M / n, 0)
            cv2.rectangle(img, (x0, self.hud_h), (x1, self.height),
                          COLOR_GRASS_DARK if i % 2 == 0 else COLOR_GRASS_LIGHT, -1)

        cv2.rectangle(img, p(0, 0), p(PITCH_W_M, PITCH_H_M), lc, 2)
        cv2.line(img, p(52.5, 0), p(52.5, PITCH_H_M), lc, 2)
        cv2.circle(img, p(52.5, 34.0), int(9.15 * s), lc, 2)
        cv2.circle(img, p(52.5, 34.0), 3, lc, -1)
        for x0, x1 in [(0, 16.5), (88.5, PITCH_W_M)]:            # vòng cấm
            cv2.rectangle(img, p(x0, 13.84), p(x1, 54.16), lc, 2)
        for x0, x1 in [(0, 5.5), (99.5, PITCH_W_M)]:             # vòng 5m50
            cv2.rectangle(img, p(x0, 24.84), p(x1, 43.16), lc, 2)
        cv2.ellipse(img, p(11.0, 34.0), (int(9.15 * s),) * 2, 0, -53, 53, lc, 2)
        cv2.ellipse(img, p(94.0, 34.0), (int(9.15 * s),) * 2, 0, 127, 233, lc, 2)
        cv2.circle(img, p(11.0, 34.0), 3, lc, -1)
        cv2.circle(img, p(94.0, 34.0), 3, lc, -1)
        cv2.rectangle(img, p(0, 30.34), p(-2, 37.66), lc, 2)     # khung thành
        cv2.rectangle(img, p(PITCH_W_M, 30.34), p(PITCH_W_M + 2, 37.66), lc, 2)
        cv2.rectangle(img, (0, 0), (self.width, self.hud_h), COLOR_HUD_BG, -1)
        return img

    def _update_trails(self, positions: Dict[int, Tuple[float, float]]) -> None:
        for tid, xy in positions.items():
            tr = self.trails.setdefault(tid, deque(maxlen=self.trail_len))
            if tr and np.hypot(xy[0] - tr[-1][0], xy[1] - tr[-1][1]) > MAX_STEP_M:
                tr.clear()                                       # ID nhảy chỗ → không nối vệt
            tr.append(xy)
        for tid in list(self.trails):                            # track biến mất → xóa vệt
            if tid not in positions:
                del self.trails[tid]

    def render(self, positions: Dict[int, Tuple[float, float]],
               title: str = "", info: str = "", progress: float = 0.0) -> np.ndarray:
        self._update_trails(positions)
        img = self.template.copy()

        overlay = img.copy()
        for tid, pts in self.trails.items():
            px = [self.to_px(*np.clip(xy, 0, [PITCH_W_M, PITCH_H_M])) for xy in pts]
            for i in range(1, len(px)):
                cv2.line(overlay, px[i - 1], px[i], color_of(tid), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        r = max(5, int(0.75 * self.scale))
        for tid, (x, y) in positions.items():
            c = self.to_px(min(max(x, 0.0), PITCH_W_M), min(max(y, 0.0), PITCH_H_M))
            cv2.circle(img, c, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, color_of(tid), -1, cv2.LINE_AA)
            label = str(tid)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, label, (c[0] - tw // 2, c[1] - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, title, (12, 22), font, 0.6, (160, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(img, info, (12, 44), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        bar_w = self.width - 24
        cv2.rectangle(img, (12, self.hud_h - 6), (12 + bar_w, self.hud_h - 3), (50, 70, 50), -1)
        cv2.rectangle(img, (12, self.hud_h - 6), (12 + int(bar_w * progress), self.hud_h - 3), (80, 210, 80), -1)
        return img


# ── Khung hình camera ─────────────────────────────────────────────────────────

def draw_tracks(frame: np.ndarray, boxes: Dict[int, List[float]], thickness: int = 3,
                font_scale: float = 1.0) -> np.ndarray:
    """Vẽ bbox + ID lên frame (in-place). boxes: {tid: [x1, y1, x2, y2]}."""
    for tid, box in boxes.items():
        x1, y1, x2, y2 = map(int, box)
        c = color_of(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness)
        label = f"{tid}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ty - th - bl - 2), (x1 + tw + 6, ty + 2), c, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 3, ty - bl), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def boxes_at(all_tracks: Dict, frame_idx: int) -> Dict[int, List[float]]:
    """{tid: tlbr} của các track có box ở frame_idx."""
    out = {}
    for tid, data in all_tracks.items():
        b = data["boxes"]
        if frame_idx < len(b) and b[frame_idx] is not None:
            out[tid] = b[frame_idx]
    return out


def side_by_side(camera: np.ndarray, pitch: np.ndarray, label: str = "") -> np.ndarray:
    """Camera (thu nhỏ theo bề rộng sơ đồ sân) ở trên, sơ đồ sân ở dưới."""
    w = pitch.shape[1]
    h = int(round(camera.shape[0] * w / camera.shape[1]))
    cam = cv2.resize(camera, (w, h), interpolation=cv2.INTER_AREA)
    if label:
        cv2.rectangle(cam, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(cam, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([cam, pitch])


def open_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """VideoWriter .mp4 (mp4v) — mở được bằng VLC / Windows Media Player."""
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not wr.isOpened():
        raise IOError(f"Không tạo được video: {path}")
    return wr
