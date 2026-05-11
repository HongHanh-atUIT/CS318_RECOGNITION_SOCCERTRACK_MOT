import sys
import os

sys.path.append(os.path.abspath(".."))

import cv2
import os
import numpy as np
from typing import List, Dict, Optional
from feature_extractor import FeatureExtractor
from detector import Detector
from tracker import Tracker
from gtalink import GTALink
from compute_metrics import compute_metrics

def process_players(frame, frame_id, player_detector,
                    player_tracker, extractor):

    detections = player_detector.detect(frame)

    enriched = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:-1])
        score = float(det[4])

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        feat = extractor.extract(crop) \
            if extractor is not None else None

        enriched.append({
            'tlbr': [float(x1), float(y1), float(x2), float(y2)],
            'score': score,
            'feat': feat
        })

    tracks = player_tracker.update(enriched, frame_id)

    return tracks

def process_balls(frame, frame_id,
                  ball_detector, ball_tracker):

    detections = ball_detector.detect(frame)

    enriched = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:-1])
        score = float(det[4])

        enriched.append({
            'tlbr': [float(x1), float(y1), float(x2), float(y2)],
            'score': score,
            'feat': None
        })

    tracks = ball_tracker.update(enriched, frame_id)

    return tracks

def update_all_tracks(
    all_tracks,
    active_tracks,
    frame_idx,
    frame=None,
    refiner=None,
    extractor_refine=None
):
    """
    Lưu tracking result vào all_tracks
    """

    active_ids = set()

    for track in active_tracks:

        tid = track.track_id
        active_ids.add(tid)

        # Track mới
        if tid not in all_tracks:

            entry = {
                'boxes': [None] * frame_idx,
                'frames': [],
            }

            if refiner is not None:
                entry['feats'] = []

            all_tracks[tid] = entry

        # Append box
        box = track.tlbr.tolist()

        all_tracks[tid]['boxes'].append(box)
        all_tracks[tid]['frames'].append(frame_idx)

        # Save feature cho refinement
        if refiner is not None:

            x1, y1, x2, y2 = map(int, track.tlbr)

            crop = frame[y1:y2, x1:x2]

            feat = (
                extractor_refine.extract(crop)
                if crop.size > 0
                else extractor_refine.extract(frame[0:1, 0:1])
            )

            all_tracks[tid]['feats'].append(feat)

    # Track inactive tại frame này
    for tid, data in all_tracks.items():

        if tid not in active_ids:

            if len(data['boxes']) == frame_idx:
                data['boxes'].append(None)


def run_mot(
    video_path,
    ball_detector,
    player_detector,
    ball_tracker,
    player_tracker,
    extractor=None,
    refiner=None,
    extractor_refine=None
):
    """
    Chạy full MOT pipeline cho player + ball
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    # Track player
    all_player_tracks: Dict[int, Dict] = {}

    # Track ball
    all_ball_tracks: Dict[int, Dict] = {}

    frame_idx = 0
    frame_id = 1

    # extractor cho refinement
    extractor_refine = (
        extractor_refine
        if (refiner is not None and extractor_refine is not None)
        else extractor
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}")

        # =====================================================
        # PLAYER PIPELINE
        # =====================================================
        active_player_tracks = process_players(
            frame,
            frame_id,
            player_detector,
            player_tracker,
            extractor
        )

        # =====================================================
        # BALL PIPELINE
        # =====================================================
        active_ball_tracks = process_balls(
            frame,
            frame_id,
            ball_detector,
            ball_tracker
        )

        # =====================================================
        # SAVE PLAYER TRACKS
        # =====================================================
        update_all_tracks(
            all_player_tracks,
            active_player_tracks,
            frame_idx,
            frame,
            refiner,
            extractor_refine
        )

        # =====================================================
        # SAVE BALL TRACKS
        # =====================================================
        update_all_tracks(
            all_ball_tracks,
            active_ball_tracks,
            frame_idx
        )

        frame_idx += 1
        frame_id += 1

    cap.release()

    print(f"\n[MOT] Done — {frame_idx} frames")
    print(f"Players: {len(all_player_tracks)} tracks")
    print(f"Balls  : {len(all_ball_tracks)} tracks")

    # =========================================================
    # REFINEMENT
    # =========================================================
    if refiner is not None:
        all_player_tracks = refiner.refine(all_player_tracks)
        all_ball_tracks = refiner.refine(all_ball_tracks)

    return {
        'players': all_player_tracks,
        'balls': all_ball_tracks
    }

def visualize_tracks(video_path: str,
                     all_tracks: Dict,
                     output_path: str = "output_tracked.avi",  # ← Dùng .avi
                     show_id: bool = True,
                     thickness: int = 2,
                     font_scale: float = 0.7):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Buộc dùng AVI + MJPG (tương thích cao nhất với Windows)
    if not output_path.lower().endswith('.avi'):
        output_path = output_path.rsplit('.', 1)[0] + '.avi'

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')      # Codec dễ mở nhất
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("⚠️ MJPG không hoạt động, thử XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Xây frame map
    frame_map: Dict[int, Dict[int, List]] = {}
    for tid, data in all_tracks.items():
        for fi, box in enumerate(data.get('boxes', [])):
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
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y = max(y1 - 5, th + 5)

                cv2.rectangle(frame, 
                              (x1, label_y - th - baseline - 4),
                              (x1 + tw + 6, label_y + 2), 
                              color, cv2.FILLED)
                
                cv2.putText(frame, label, (x1 + 3, label_y - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        cv2.putText(frame, f"Frame {frame_idx+1}/{total}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Đã lưu xong: {output_path}")
    
    
def run_pipeline(
    video_path:        str,
    gt_csv_path:       str,
    detector,
    tracker_name:      str,
    output_path = None,
    extractor=None,                # online extractor (DeepEIoU dùng, ByteTrack bỏ qua)
    refiner=None,
    refiner_extractor=None,        # extractor cho GTALink
    iou_thresh:        float = 0.5,
    tracker_kwargs:    Optional[Dict] = None,
    visualize = False,
    **viz_kwargs
) -> Dict:
    """
    Chạy full pipeline MOT và tính metrics.

    Parameters
    ----------
    video_path        : đường dẫn video
    gt_csv_path       : đường dẫn file annotation CSV (để tính metrics)
    detector          : Detector instance
    tracker_name      : tên tracker ("deepeiou" | "bytetrack" | "ocsort" | "strongsort")
    output_path       : đường dẫn lưu video output
    extractor         : extractor cho online tracking (None nếu tracker không dùng)
    refiner           : GTALink instance (None nếu không dùng refinement)
    refiner_extractor : extractor cho GTALink (None nếu không dùng refinement)
    iou_thresh        : IoU threshold để tính metrics
    tracker_kwargs    : kwargs truyền vào tracker constructor

    Returns
    -------
    dict với all_tracks và metrics
    """
    if tracker_name in ('strongsort', 'deepeiou') and extractor is None:
        raise ValueError(
            f"Tracker '{tracker_name}' yêu cầu appearance extractor. "
            f"Vui lòng truyền extractor != None."
        )
    tracker_kwargs = tracker_kwargs or {}
    tracker        = Tracker(algorithm=tracker_name, **tracker_kwargs)
    ext_name = extractor.backend if extractor is not None else 'None'
    ref_name = refiner_extractor.backend if refiner_extractor is not None else \
               (extractor.backend if extractor is not None else 'None')
 
    print(f"\n{'='*60}")
    print(f"PIPELINE: {detector.backend.upper()} → "
          f"{ext_name.upper()} → "
          f"{tracker_name.upper()} → "
          f"GTALink({ref_name})")
    print(f"{'='*60}")

    # Bước 1-3: MOT
    all_tracks = run_mot(
        video_path      = video_path,
        detector        = detector,
        tracker         = tracker,
        extractor       = extractor,
        refiner         = refiner,
        extractor_refine= refiner_extractor,
    )

    # Bước 4: Visualize
    visualize_tracks(
        video_path  = video_path,
        all_tracks  = all_tracks,
        output_path = output_path,
        **viz_kwargs,
    )

    # Tính metrics
    metrics = compute_metrics(
        all_tracks = all_tracks,
        csv_path   = gt_csv_path,
        iou_thresh = iou_thresh,
    )

    return {'all_tracks': all_tracks, 'metrics': metrics}