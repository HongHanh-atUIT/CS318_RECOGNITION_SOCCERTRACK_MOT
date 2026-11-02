"""
demo.py — E2ESoTrack end-to-end: video thô → tracking → tọa độ sân → video 2D.

    [1] Detection      : YOLO26 (conf 0.4, iou 0.5, imgsz 1600)
    [2] Online tracking: ByteTrack (use_project=False, lambda_iou=1.0)
    [3] Offline refine : GTALink + SOLIDER (merge_dist_thres=0.35)   — chỉ ở cấu hình "optimal"
    [4] Pitch localize : homography từ fisheye_keypoints.json, điểm chân = bottom-center bbox
    [5] Render         : video camera có ID, sơ đồ sân 2D, video ghép, CSV quỹ đạo
    [6] Evaluate       : MOTA/IDF1/HOTA (nếu có GT) và Pitch-HOTA/E_1to1 (nếu clip có GNSS)

Cấu hình và hàm gọi giống hệt lúc chạy thực nghiệm (test.ipynb / eval_pitch.py),
nên số liệu in ra trùng với bảng kết quả.

Ví dụ:
    python demo.py --video ../Data/wide_view/videos/F_20220220_1_0720_0750.mp4
    python demo.py --video <video.mp4> --config fast --out ../Output/demo_fast
    python demo.py --video <video.mp4> --tracks ../Output/demo/tracks_refined.pkl   # chỉ render lại
"""

import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.pitch_localization import load_homography
from src.visualize_2d import PitchCanvas, boxes_at, draw_tracks, open_writer, side_by_side, tracks_to_pitch

BASE         = os.path.dirname(HERE)
DATA         = os.path.join(BASE, "Data")
MODELS       = os.path.join(HERE, "models")
YOLO_PATH    = os.path.join(MODELS, "best_yolo26.pt")
SOLIDER_PATH = os.path.join(MODELS, "swin_small_converted.pth")
KEYPOINTS    = os.path.join(DATA, "fisheye_keypoints.json")
GT_DIR       = os.path.join(DATA, "wide_view", "annotations")
GNSS_DIR     = os.path.join(DATA, "GNSS")
GNSS_KP      = os.path.join(DATA, "gnss_keypoints.json")

TRACKER_KWARGS = {"use_project": False, "lambda_iou": 1.0}   # như test.ipynb
GTA_KWARGS     = {"merge_dist_thres": 0.35}                   # như test.ipynb
PIPELINE_NAME  = {"fast":    "YOLO26 > ByteTrack",
                  "optimal": "YOLO26 > ByteTrack > GTALink(SOLIDER)"}


# ── Các bước pipeline ─────────────────────────────────────────────────────────

def build_components(config: str, yolo_path: str = YOLO_PATH, solider_path: str = SOLIDER_PATH):
    """Detector YOLO26, và SOLIDER cho GTALink nếu config == 'optimal'."""
    from src.detector import Detector
    detector = Detector(backend="yolo26", model_path=yolo_path)
    extractor_refine = None
    if config == "optimal":
        from src.feature_extractor import FeatureExtractor
        extractor_refine = FeatureExtractor(backend="solider", solider_model_path=solider_path,
                                            solider_arch="swin_small", solider_semantic_weight=0.2)
    return detector, extractor_refine


def track_online(video_path: str, detector, extractor_refine=None):
    """Detection + ByteTrack từng frame. Nếu có extractor_refine thì lưu kèm embedding cho GTALink."""
    from src.tracker import Tracker
    from src.utils import run_mot
    tracker = Tracker(algorithm="bytetrack", **TRACKER_KWARGS)
    all_tracks, normal_time, extract_time = run_mot(video_path, detector, tracker, extractor=None,
                                                    refiner=None, extractor_refine=extractor_refine)
    return all_tracks, normal_time, extract_time


def refine_offline(all_tracks: dict) -> dict:
    """GTALink: tách tracklet bị lẫn ID (DBSCAN) rồi nối tracklet cùng người."""
    from src.gtalink import GTALink
    return GTALink(**GTA_KWARGS).refine(all_tracks)


def strip_feats(all_tracks: dict) -> dict:
    return {tid: {"boxes": d["boxes"], "frames": d["frames"]} for tid, d in all_tracks.items()}


def render(video_path: str, all_tracks: dict, H: np.ndarray, out_dir: str, title: str,
           gif_path: str = None, gif_frames: int = 250, gif_width: int = 640) -> dict:
    """Ghi tracked_camera.mp4, pitch_2d.mp4, side_by_side.mp4, trajectories.csv (+ GIF tùy chọn)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    positions = tracks_to_pitch(all_tracks, H)
    canvas    = PitchCanvas(width=1280)
    cam_size  = (fw // 2 // 2 * 2, fh // 2 // 2 * 2)                  # camera thu nửa, kích thước chẵn
    paths = {k: os.path.join(out_dir, f"{k}.mp4") for k in ("tracked_camera", "pitch_2d", "side_by_side")}
    w_cam = w_pitch = w_sbs = None
    gif = []

    rows, f = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = boxes_at(all_tracks, f)
        pos   = positions.get(f, {})
        for tid, (x1, y1, x2, y2) in boxes.items():
            xy = pos.get(tid, (np.nan, np.nan))
            rows.append(f"{f},{f / fps:.3f},{tid},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{xy[0]:.3f},{xy[1]:.3f}")

        cam   = cv2.resize(draw_tracks(frame, boxes, thickness=6, font_scale=1.6), cam_size, interpolation=cv2.INTER_AREA)
        pitch = canvas.render(pos, title=title,
                              info=f"Frame {f + 1}/{total}   t = {f / fps:5.2f} s   Players: {len(pos)}",
                              progress=(f + 1) / max(total, 1))
        sbs   = side_by_side(cam, pitch, label="Wide-angle camera  ->  2D tactical map (homography)")
        sbs   = sbs[: sbs.shape[0] // 2 * 2]

        if w_cam is None:
            w_cam   = open_writer(paths["tracked_camera"], fps, (cam.shape[1], cam.shape[0]))
            w_pitch = open_writer(paths["pitch_2d"], fps, (pitch.shape[1], pitch.shape[0]))
            w_sbs   = open_writer(paths["side_by_side"], fps, (sbs.shape[1], sbs.shape[0]))
        w_cam.write(cam)
        w_pitch.write(pitch)
        w_sbs.write(sbs)

        if gif_path and f < gif_frames and f % 3 == 0:
            g = cv2.resize(sbs, (gif_width, int(sbs.shape[0] * gif_width / sbs.shape[1])), interpolation=cv2.INTER_AREA)
            gif.append(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))
        if f % 150 == 0:
            print(f"  render frame {f}/{total}")
        f += 1

    cap.release()
    for w in (w_cam, w_pitch, w_sbs):
        if w is not None:
            w.release()

    paths["trajectories"] = os.path.join(out_dir, "trajectories.csv")
    with open(paths["trajectories"], "w", encoding="utf-8") as fp:
        fp.write("frame,time_s,track_id,x1,y1,x2,y2,x_m,y_m\n")
        fp.write("\n".join(rows) + "\n")

    if gif_path and gif:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, gif, duration=1000 * 3 / fps, loop=0)
        paths["gif"] = gif_path
    return paths


def evaluate(video_path: str, all_tracks: dict, H: np.ndarray) -> dict:
    """MOT metrics nếu có annotation; Pitch-HOTA / E_1to1 nếu clip nằm trong vùng có GNSS."""
    import contextlib
    import io
    name = os.path.splitext(os.path.basename(video_path))[0]
    gt_csv = os.path.join(GT_DIR, name + ".csv")
    out = {}
    if not os.path.exists(gt_csv):
        return out

    from src.compute_metrics import compute_metrics
    with contextlib.redirect_stdout(io.StringIO()):
        m = compute_metrics(all_tracks, gt_csv, verbose=False)
    out.update({k: float(m[k]) for k in ("MOTA", "IDF1", "HOTA", "IDs")})

    if os.path.isdir(GNSS_DIR) and os.path.exists(GNSS_KP):
        from src.pitch_eval import GNSSGroundTruth, clip_index, evaluate_clip
        gnss = GNSSGroundTruth(GNSS_DIR, GNSS_KP)
        if clip_index(video_path) < len(os.listdir(GNSS_DIR)):          # mỗi file GNSS = 1 clip 30 s
            r = evaluate_clip(all_tracks, gt_csv, os.path.basename(video_path), gnss, H)
            out.update({"Pitch-HOTA": float(r["Pitch-HOTA"]), "E_1to1": float(r["E_1to1"])})
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="E2ESoTrack demo: raw video → 2D tactical map video")
    ap.add_argument("--video", required=True, help="video wide-view (6500×1000)")
    ap.add_argument("--config", choices=["fast", "optimal"], default="optimal")
    ap.add_argument("--out", default=os.path.join(BASE, "Output", "demo"))
    ap.add_argument("--tracks", default=None, help="pickle all_tracks có sẵn → bỏ qua bước 1-3")
    ap.add_argument("--gif", action="store_true", help="xuất thêm demo.gif (10 s đầu) cho README/slide")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    title = PIPELINE_NAME[args.config]
    print(f"\n=== E2ESoTrack demo | {title} | {os.path.basename(args.video)} ===")
    H, _ = load_homography(KEYPOINTS)
    timing = {}

    if args.tracks:
        with open(args.tracks, "rb") as f:
            tracks = pickle.load(f)
        print(f"[1-3] Dùng tracks có sẵn: {args.tracks} ({len(tracks)} IDs)")
    else:
        print("\n[1/6] Load models")
        detector, extractor_refine = build_components(args.config)

        print("\n[2/6] Detection + ByteTrack (online)")
        online, timing["detect+track_s"], timing["extract_s"] = track_online(args.video, detector, extractor_refine)
        with open(os.path.join(args.out, "tracks_online.pkl"), "wb") as f:
            pickle.dump(strip_feats(online), f)
        tracks = online

        if args.config == "optimal":
            print("\n[3/6] GTALink (offline refinement)")
            t0 = time.time()
            tracks = refine_offline(online)
            timing["gta_s"] = time.time() - t0
            print(f"  {len(online)} → {len(tracks)} IDs")
        else:
            print("\n[3/6] (fast) bỏ qua GTALink")
        tracks = strip_feats(tracks)
        with open(os.path.join(args.out, "tracks_refined.pkl" if args.config == "optimal" else "tracks_fast.pkl"), "wb") as f:
            pickle.dump(tracks, f)

    print("\n[4-5/6] Pitch localization + render video 2D")
    paths = render(args.video, tracks, H, args.out, title,
                   gif_path=os.path.join(args.out, "demo.gif") if args.gif else None)

    metrics = {}
    if not args.no_eval:
        print("\n[6/6] Evaluate")
        metrics = evaluate(args.video, tracks, H)
        for k, v in metrics.items():
            print(f"  {k:>10}: {v:.2f}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"video": os.path.basename(args.video), "config": args.config, "pipeline": title,
                   "n_ids": len(tracks), "timing": timing, "metrics": metrics, "outputs": paths}, f, indent=2)
    print("\nOutputs:")
    for k, p in paths.items():
        print(f"  {k:>15}: {p}")


if __name__ == "__main__":
    main()
