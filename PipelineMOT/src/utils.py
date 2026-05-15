from src.compute_metrics import compute_metrics
from typing import List, Dict, Optional
from src.tracker import Tracker
import time
import cv2

# Plot 
_PALETTE = [
    (255, 99, 71),    # Tomato
    (255, 165, 0),    # Orange
    (255, 215, 0),    # Gold
    (154, 205, 50),   # YellowGreen
    (50, 205, 50),    # LimeGreen
    (0, 255, 127),    # SpringGreen
    (0, 206, 209),    # DarkTurquoise
    (30, 144, 255),   # DodgerBlue
    (65, 105, 225),   # RoyalBlue
    (138, 43, 226),   # BlueViolet
    (186, 85, 211),   # MediumOrchid
    (255, 20, 147),   # DeepPink
    (255, 105, 180),  # HotPink
    (210, 180, 140),  # Tan
    (244, 164, 96),   # SandyBrown
    (46, 139, 87),    # SeaGreen
    (70, 130, 180),   # SteelBlue
    (123, 104, 238),  # MediumSlateBlue
    (199, 21, 133),   # MediumVioletRed
    (255, 69, 0),     # OrangeRed
    (0, 191, 255),    # DeepSkyBlue
    (127, 255, 212),  # Aquamarine
]

def _color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


def visualize_tracks(video_path: str,
                     all_tracks: Dict,
                     output_path: str = "output_tracked.avi",
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

    if not output_path.lower().endswith('.avi'):
        output_path = output_path.rsplit('.', 1)[0] + '.avi'

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("⚠️ MJPG không hoạt động, thử XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
    print(f"✅ Đã lưu xong: {output_path}")
    print("   → Hãy thử mở file này bằng Windows Media Player")

    
def refine(all_tracks, refiner):
    return refiner.refine(all_tracks)

        
def run_mot(video_path, detector, tracker, extractor=None, refiner=None, extractor_refine=None):
    """
    Chạy full MOT pipeline cho một video.
 
    Parameters
    ----------
    video_path       : str
    detector         : object với method .detect(frame) → [[x1,y1,x2,y2,conf], ...]
    tracker          : Tracker instance
    extractor        : object với method .extract_batch(crops) → list[np.ndarray]
                       dùng cho online tracking
    refiner          : (optional) object với .refine(all_tracks) → refined_tracks
    extractor_refine : (optional) object với method .extract_batch(crops) → list[np.ndarray]
                       Extractor riêng để tạo embedding cho refinement.
 
    Returns
    -------
    all_tracks : dict
        {
            track_id (int): {
                'boxes' : list — box [x1,y1,x2,y2] hoặc None theo từng frame,
                'frames': list — frame_idx tương ứng (chỉ các frame có box)
                'feats' : list — feature tương ứng (nếu dùng refiner)
            }
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    all_tracks: Dict[int, Dict] = {}

    frame_idx = 0
    frame_id  = 1

    need_refine_feat = extractor_refine is not None
    total_extract = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx}")

        # ── Bước 2.1: Detect ────────────────────────────────────────────
        detections = detector.detect(frame)  # [[x1, y1, x2, y2, conf], ...]

        # ── Bước 2.2: Crop tất cả detection trong frame ─────────────────
        valid_dets  = []   # detection sau khi lọc crop rỗng
        det_crops   = []   # crops tương ứng (cho online extractor)

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:-1])
            score = float(det[4])
            crop  = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            valid_dets.append({'tlbr': [float(x1), float(y1), float(x2), float(y2)],
                                'score': score})
            det_crops.append(crop)

        # ── Bước 2.3: Batch feature extraction (online) ─────────────────
        if extractor is not None and det_crops:
            online_feats = extractor.extract_batch(det_crops)   # list[np.ndarray]
        else:
            online_feats = [None] * len(valid_dets)

        enriched_detections = []
        for det_info, feat in zip(valid_dets, online_feats):
            enriched_detections.append({**det_info, 'feat': feat})

        # ── Bước 2.4: Tracking ──────────────────────────────────────────
        active_tracks = tracker.update(enriched_detections, frame_id)

        # ── Bước 2.5: Batch feature extraction (refinement) ─────────────
        # Chỉ crop lại từ active tracks (có thể khác với detection crops
        # do tracker dùng Kalman smoothing / predicted box).
        
        start_extract = time.time()
        if need_refine_feat and active_tracks:
            refine_crops = []
            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.tlbr)
                crop = frame[y1:y2, x1:x2]
                # fallback nếu crop rỗng (box nằm sát biên)
                refine_crops.append(crop if crop.size > 0 else frame[0:1, 0:1])

            refine_feats = extractor_refine.extract_batch(refine_crops)  # list[np.ndarray]
        else:
            refine_feats = [None] * len(active_tracks)
        end_extract = time.time()
        total_extract += end_extract - start_extract
        
        # ── Bước 2.6: Lưu all_tracks ────────────────────────────────────
        active_ids = set()
        for track, ref_feat in zip(active_tracks, refine_feats):
            tid = track.track_id
            active_ids.add(tid)

            if tid not in all_tracks:
                entry = {
                    'boxes' : [None] * frame_idx,
                    'frames': [],
                }
                if need_refine_feat:
                    entry['feats'] = []
                all_tracks[tid] = entry

            box = track.tlbr.tolist()
            all_tracks[tid]['boxes'].append(box)
            all_tracks[tid]['frames'].append(frame_idx)

            if need_refine_feat:
                all_tracks[tid]['feats'].append(ref_feat)

        # Track đã có nhưng không active frame này → None
        for tid, data in all_tracks.items():
            if tid not in active_ids:
                if len(data['boxes']) == frame_idx:
                    data['boxes'].append(None)

        frame_idx += 1
        frame_id  += 1

    cap.release()
    end = time.time()
    total = end - start
    normal = total - total_extract
    
    print(f"\n[MOT] Done — {frame_idx} frames, {len(all_tracks)} tracks")
    print(f"[MOT] Normal time:{(normal):.2f} seconds, extract (refiner) time: {total_extract:.2f} seconds")
    return all_tracks


def run_pipeline(
    video_path:        str,
    gt_csv_path:       str,
    detector,
    tracker_name:      str,
    output_path:       str,
    extractor=None,
    input_tracks=None,
    refiner=None,
    refiner_extractor=None,
    iou_thresh:        float = 0.5,
    tracker_kwargs:    Optional[Dict] = None,
    visualize=False,
    **viz_kwargs
) -> Dict:
    """
    Chạy full pipeline MOT và tính metrics cho 1 video.

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
    visualize         : Nếu True -> lưu video visualize
    input_tracks      : kết quả MOT sẵn có (bỏ qua bước chạy lại pipeline)

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
    ref_name = refiner_extractor.backend if refiner_extractor is not None else \
               (extractor.backend if extractor is not None else 'None')

    if input_tracks is None:
        print("===== Chưa có kết quả MOT - Chạy toàn bộ pipeline ======")
        all_tracks = run_mot(
            video_path       = video_path,
            detector         = detector,
            tracker          = tracker,
            extractor        = extractor,
            refiner          = refiner,
            extractor_refine = refiner_extractor,
        )
        if refiner is not None:
            print(f"[OFFLINE TRACKER] Đang refine output bằng GTALink với extractor là {ref_name}")
            all_tracks = refine(all_tracks, refiner)
    else:
        print("===== Đã có kết quả MOT - Tái sử dụng ======")
        start = time.time()
        if refiner is not None:
            print(f"[OFFLINE TRACKER] Đang refine output bằng GTALink với extractor là {ref_name}")
            all_tracks = refine(input_tracks, refiner)
            end = time.time()
            print(f"[OFFLINE TRACKER] Hoàn tất sau {(end-start):.2f} seconds.")
        else:
            all_tracks = input_tracks

    if visualize:
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

    return {'all_tracks': all_tracks, 'metrics': metrics}