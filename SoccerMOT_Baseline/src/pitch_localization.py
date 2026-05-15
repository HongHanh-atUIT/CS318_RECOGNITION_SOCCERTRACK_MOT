"""
pitch_localization.py
---------------------
Ánh xạ bounding box ↔ tọa độ pitch qua homography (hoặc normalize).

  load_homography           : load keypoints JSON → tính H, H_inv
  box_to_pitch_xy           : bbox pixel  →  [x, y] pitch (m)
  pitch_xy_to_bottom_center : [x, y] pitch →  bottom-center pixel
  tlwh_from_pitch_xy        : pitch [x,y] + last_wh  →  tlwh pixel

NOTE về frame_w / frame_h:
  - Chỉ dùng khi H is None (normalize mode).
  - Khi H is not None, homography tự xử lý tọa độ pixel → pitch.
  - Tuy nhiên vẫn phải truyền đúng frame_w/frame_h vào STrack để
    euclidean_distance fallback (khi mean is None) hoạt động đúng.
  - Video wide_view có kích thước 6500x1000 — KHÔNG phải 1920x1080.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple


# ── Homography loader ──────────────────────────────────────────────────────────

def load_homography(
    keypoints_path: str,
    ransac_thresh:  float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load file fisheye_keypoints.json và tính homography H (pixel → pitch).

    File JSON có dạng:
        { "(x_pitch, y_pitch)": [pixel_x, pixel_y], ... }

    Returns
    -------
    H     : (3,3) float64  — pixel → pitch (dùng trong box_to_pitch_xy)
    H_inv : (3,3) float64  — pitch → pixel (dùng trong pitch_xy_to_bottom_center)
    """
    with open(keypoints_path, "r") as f:
        raw = json.load(f)

    src_pts, dst_pts = [], []
    for key, val in raw.items():
        x_pitch, y_pitch = eval(key)          # tọa độ sân (m)
        pixel_x, pixel_y = float(val[0]), float(val[1])
        src_pts.append([pixel_x, pixel_y])    # pixel trong ảnh
        dst_pts.append([x_pitch, y_pitch])    # điểm tương ứng trên sân

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError(
            f"findHomography thất bại — kiểm tra lại file {keypoints_path}"
        )

    inliers = int(mask.ravel().sum())
    print(f"[pitch_localization] H computed — inliers: {inliers}/{len(src_pts)}")

    H_inv = np.linalg.inv(H)
    return H.astype(np.float64), H_inv.astype(np.float64)


# ── Core functions ─────────────────────────────────────────────────────────────

def box_to_pitch_xy(
    tlwh:    np.ndarray,
    H:       Optional[np.ndarray] = None,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> np.ndarray:
    """
    Bounding box [x1,y1,w,h] (pixel) → [x, y] trên pitch (m).

    Điểm chân cầu thủ = bottom-center:
        bx = x1 + w/2
        by = y1 + h

    H is None  → normalize: [bx/frame_w, by/frame_h]  ∈ [0,1]²
    H not None → homography: H @ [bx, by, 1]ᵀ  rồi chia w → (x_m, y_m)

    NOTE: frame_w/frame_h chỉ dùng khi H is None.
          Khi có H, homography tự map pixel → pitch không cần normalize.
    """
    x1, y1, w, h = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
    bx = x1 + w / 2.0
    by = y1 + h                              # chân cầu thủ

    if H is None:
        return np.array([bx / frame_w, by / frame_h], dtype=np.float64)

    pt  = np.array([bx, by, 1.0], dtype=np.float64)
    dst = H @ pt
    dst /= (dst[2] + 1e-9)
    return dst[:2].astype(np.float64)        # [x_m, y_m] trên sân


def pitch_xy_to_bottom_center(
    xy:      np.ndarray,
    H_inv:   Optional[np.ndarray] = None,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> np.ndarray:
    """
    [x, y] pitch (m) → bottom-center pixel [bx, by].

    H_inv is None  → de-normalize: [x*frame_w, y*frame_h]
    H_inv not None → H_inv @ [x, y, 1]ᵀ  rồi chia w
    """
    if H_inv is None:
        return np.array([xy[0] * frame_w, xy[1] * frame_h], dtype=np.float64)

    pt  = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    dst = H_inv @ pt
    dst /= (dst[2] + 1e-9)
    return dst[:2].astype(np.float64)        # [pixel_x, pixel_y]


def tlwh_from_pitch_xy(
    xy:      np.ndarray,
    last_wh: np.ndarray,
    H_inv:   Optional[np.ndarray] = None,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> np.ndarray:
    """
    Recover tlwh [x1,y1,w,h] từ pitch [x,y] + kích thước box cuối.
    Dùng trong STrack.tlwh khi use_project=True.
    """
    bx, by = pitch_xy_to_bottom_center(xy, H_inv, frame_w, frame_h)
    w, h   = float(last_wh[0]), float(last_wh[1])
    x1 = bx - w / 2.0
    y1 = by - h
    return np.array([x1, y1, w, h], dtype=np.float64)


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    kp_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\UITs subject\Năm 3\Nhận dạng\CS318_RECOGNITION_SOCCERTRACK_MOT\Data\fisheye_keypoints.json"

    if not Path(kp_path).exists():
        print(f"File không tìm thấy: {kp_path}")
        sys.exit(1)

    H, H_inv = load_homography(kp_path)

    with open(kp_path) as f:
        raw = json.load(f)

    errors = []
    for key, val in raw.items():
        x_pitch, y_pitch = eval(key)
        pixel_x, pixel_y = float(val[0]), float(val[1])

        pt = np.array([pixel_x, pixel_y, 1.0])
        proj = H @ pt
        proj /= proj[2]
        err = np.sqrt((proj[0] - x_pitch)**2 + (proj[1] - y_pitch)**2)
        errors.append(err)

    errors = np.array(errors)
    print(f"\nReprojection error (pixel→pitch):")
    print(f"  mean  : {errors.mean():.4f} m")
    print(f"  max   : {errors.max():.4f} m")
    print(f"  median: {np.median(errors):.4f} m")