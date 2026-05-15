"""
utils.py — YOLOv5 + OSNet + DeepSort pipeline.
Giữ nguyên signature run_pipeline() để tương thích notebook.
GTALink đã loại bỏ (refiner bị ignore).
"""

import cv2
import numpy as np
from typing import Dict, Optional

from .compute_metrics import compute_metrics


_PALETTE = [
    (255,99,71),(255,165,0),(255,215,0),(154,205,50),(50,205,50),
    (0,255,127),(0,206,209),(30,144,255),(65,105,225),(138,43,226),
    (186,85,211),(255,20,147),(255,105,180),(210,180,140),(244,164,96),
    (46,139,87),(70,130,180),(123,104,238),(199,21,133),(255,69,0),
    (0,191,255),(127,255,212),
]

def _color(tid): return _PALETTE[tid % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────
# run_mot
# ─────────────────────────────────────────────────────────────

def run_mot(
    video_path,
    detector,
    tracker,
    extractor=None,
    refiner=None,           # ignored — kept for compatibility
    extractor_refine=None,  # ignored
    input_tracks=None,
) -> Dict:
    if refiner is not None:
        print("[run_mot] refiner (GTALink) bị bỏ qua.")
    if input_tracks is not None:
        print("[run_mot] Reusing cached tracks.")
        return input_tracks

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    all_tracks: Dict[int, Dict] = {}
    frame_idx = 0
    frame_id  = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\r[MOT] Processing frame {frame_idx}", end="", flush=True)

        raw_dets = detector.detect(frame)

        enriched = []
        for det in raw_dets:
            x1 = max(0, int(det[0]));  y1 = max(0, int(det[1]))
            x2 = min(frame.shape[1]-1, int(det[2]))
            y2 = min(frame.shape[0]-1, int(det[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            feat = extractor.extract(crop) if extractor is not None else None
            enriched.append({
                "tlbr":  [float(x1), float(y1), float(x2), float(y2)],
                "score": float(det[4]),
                "feat":  feat,
            })

        active_tracks = tracker.update(enriched, frame_id) if enriched else []

        active_ids = set()
        for track in active_tracks:
            tid = track.track_id
            active_ids.add(tid)
            if tid not in all_tracks:
                all_tracks[tid] = {"boxes": [None] * frame_idx, "frames": []}
            tlwh = track.tlwh
            box  = [float(tlwh[0]), float(tlwh[1]),
                    float(tlwh[0]+tlwh[2]), float(tlwh[1]+tlwh[3])]
            all_tracks[tid]["boxes"].append(box)
            all_tracks[tid]["frames"].append(frame_idx)

        for tid, data in all_tracks.items():
            if tid not in active_ids and len(data["boxes"]) == frame_idx:
                data["boxes"].append(None)

        frame_idx += 1
        frame_id  += 1

    cap.release()
    print(f"\n[MOT] Done — {frame_idx} frames, {len(all_tracks)} tracks")
    return all_tracks


# ─────────────────────────────────────────────────────────────
# visualize_tracks
# ─────────────────────────────────────────────────────────────

def visualize_tracks(
    video_path, all_tracks,
    output_path="output_tracked.avi",
    show_id=True, thickness=2, font_scale=0.6,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not output_path.lower().endswith(".avi"):
        output_path = output_path.rsplit(".", 1)[0] + ".avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*"XVID"),
                                  fps, (width, height))

    frame_map: Dict[int, Dict] = {}
    for tid, data in all_tracks.items():
        for fi, box in enumerate(data.get("boxes", [])):
            if box is not None:
                frame_map.setdefault(fi, {})[tid] = box

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for tid, box in frame_map.get(frame_idx, {}).items():
            x1, y1, x2, y2 = map(int, box)
            color = _color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if show_id:
                label = f"ID {tid}"
                (tw, th), base = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                ly = max(y1 - 5, th + 5)
                cv2.rectangle(frame, (x1, ly-th-base-4),
                              (x1+tw+6, ly+2), color, cv2.FILLED)
                cv2.putText(frame, label, (x1+3, ly-base),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255,255,255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {frame_idx+1}/{total}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Visualize] Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# run_pipeline  —  giữ nguyên signature notebook
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    video_path,
    gt_csv_path,
    detector,
    tracker_name:    str   = "deepsort",
    output_path:     str   = None,
    extractor=None,
    refiner=None,
    refiner_extractor=None,
    iou_thresh:      float = 0.5,
    tracker_kwargs:  Optional[Dict] = None,
    visualize:       bool  = False,
    input_tracks:    Optional[Dict] = None,
    **viz_kwargs,
) -> Dict:
    """
    Pipeline: YOLOv5 → OSNet → DeepSort → Metrics.
    Signature giữ nguyên để tương thích notebook cũ.
    tracker_name bị bỏ qua — luôn dùng DeepSort.
    """
    from .tracker import Tracker

    tracker_kwargs = tracker_kwargs or {}

    ext_name = extractor.backend.upper() if extractor is not None else "NONE"
    fw = tracker_kwargs.get('frame_w', '?')
    fh = tracker_kwargs.get('frame_h', '?')
    up = tracker_kwargs.get('use_project', False)

    print(f"\n{'='*60}")
    print(f"PIPELINE: {detector.backend.upper()} → {ext_name} → DEEPSORT")
    print(f"  frame_w={fw}  frame_h={fh}  use_project={up}")
    print(f"{'='*60}")

    tracker = Tracker(algorithm="deepsort", **tracker_kwargs)

    all_tracks = run_mot(
        video_path   = video_path,
        detector     = detector,
        tracker      = tracker,
        extractor    = extractor,
        refiner      = None,
        input_tracks = input_tracks,
    )

    if visualize and output_path is not None:
        visualize_tracks(
            video_path  = video_path,
            all_tracks  = all_tracks,
            output_path = output_path,
            **viz_kwargs,
        )

    metrics = compute_metrics(
        all_tracks = all_tracks,
        csv_path   = gt_csv_path,
        iou_thresh = iou_thresh,
    )

    return {"all_tracks": all_tracks, "metrics": metrics}