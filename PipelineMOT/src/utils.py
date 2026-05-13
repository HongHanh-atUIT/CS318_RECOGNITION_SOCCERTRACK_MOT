from src.compute_metrics import compute_metrics
from typing import List, Dict, Optional
from src.tracker import Tracker
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
    print(f"✅ Đã lưu xong: {output_path}")
    print("   → Hãy thử mở file này bằng Windows Media Player")
    
def refine(all_tracks, refiner):
    return refiner.refine(all_tracks)
        
    
def run_mot(video_path, detector, tracker, extractor=None, refiner = None, extractor_refine = None):
    """
    Chạy full MOT pipeline cho một video.
 
    Parameters
    ----------
    video_path : str
    detector   : object với method .detect(frame) → [[x1,y1,x2,y2,conf], ...]
    tracker    : Tracker instance
    extractor  : object với method .extract(crop) → np.ndarray (L2-normalised), dùng cho online tracking
    refiner    : (optional) object với .refine(all_tracks) → refined_tracks
                 Nếu None, trả về all_tracks trực tiếp.
    extractor_refine: (optional) object với method .extract(crop) → np.ndarray
                       Extractor riêng để tạo embedding cho refinement.
                       Nếu None nhưng refiner được truyền vào → dùng chung với online tracking
 
    Returns
    -------
    all_tracks : dict
        {
            track_id (int): {
                'boxes' : list — box [x1,y1,x2,y2] hoặc None theo từng frame,
                'frames': list — frame_idx tương ứng (chỉ các frame có box)
                'feat'  : list - feature tương ứng (nếu dùng refiner)
            }
        }
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")
    
    # all_tracks[track_id]['boxes'][i] = box tại frame i, hoặc None nếu không có
    all_tracks: Dict[int, Dict] = {} 

    frame_idx = 0   # 0-based index để index vào list
    frame_id  = 1   # 1-based id truyền vào tracker (convention của STrack)
    
    # extractor của refiner
    extractor_refine = extractor_refine if (refiner is not None and extractor_refine is not None) else extractor
    
    # Duyệt qua từng frame 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}")
        
        # Bước 2.1: Detect
        detections = detector.detect(frame) # Trả về list các box và conf [[x1, y1, x2, y2, conf]]
        
        # Bước 2.2: Crop + Feature
        enriched_detections = [] # Tổng hợp box, conf, feature tại frame đang xét của các object
        for det in detections:
            enriched = {}
            x1, y1, x2, y2 = map(int, det[:-1])
            score = float(det[4])
            
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # 2.2.1 Extract feature
            feature = extractor.extract(crop) if extractor is not None else None

            # 2.2.2 attach feature
            enriched_detections.append({
                'tlbr' : [float(x1), float(y1), float(x2), float(y2)],
                'score': score,
                'feat' : feature,
            })

        # Bước 2.3: Tracking
        active_tracks  = tracker.update(enriched_detections, frame_id) 
         
        # Lưu all_tracks
        # Với các track_id mới: chèn None cho tất cả frame trước đó
        active_ids = set()
        for track in active_tracks:
            tid = track.track_id
            active_ids.add(tid)
 
            if tid not in all_tracks:
                # Track mới → fill None cho các frame trước
                entry = {
                    'boxes' : [None] * frame_idx,
                    'frames': [],
                }
                if refiner is not None:
                    entry['feats'] = []         # chỉ khởi tạo khi cần refinement
                all_tracks[tid] = entry
 
            box = track.tlbr.tolist()   # [x1,y1,x2,y2] 
            all_tracks[tid]['boxes'].append(box)
            all_tracks[tid]['frames'].append(frame_idx)
 
            # Nếu cần refine:
            if refiner is not None:
                x1, y1, x2, y2 = map(int, track.tlbr)
                # Clamp để tránh ra ngoài biên frame
                crop = frame[y1:y2, x1:x2]
                feat = extractor_refine.extract(crop) if crop.size > 0 \
                    else extractor_refine.extract(frame[0:1, 0:1])  # fallback crop rỗng
                all_tracks[tid]['feats'].append(feat)
                    
        # Track đã có trong dict nhưng không active frame này → None
        for tid, data in all_tracks.items():
            if tid not in active_ids:
                # Chỉ thêm None nếu track chưa có entry cho frame này
                if len(data['boxes']) == frame_idx:
                    data['boxes'].append(None)
 
        frame_idx += 1
        frame_id  += 1
        
    cap.release()
    print(f"\n[MOT] Done — {frame_idx} frames, {len(all_tracks)} tracks")
    return all_tracks

def run_pipeline(
    video_path:        str,
    gt_csv_path:       str,
    detector,
    tracker_name:      str,
    output_path:       str,
    extractor=None,                # online extractor (DeepEIoU dùng, ByteTrack bỏ qua)
    input_tracks = None,             # tận dụng lại all_tracks đã chạy không refiner
    refiner=None,
    refiner_extractor=None,        # extractor cho GTALink
    iou_thresh:        float = 0.5,
    tracker_kwargs:    Optional[Dict] = None,
    visualize = False,
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
    input_tracks      : list các track là output của pipeline MOT không refiner, dùng để tối ưu thời gian, không chạy lại pipeline MOT

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
    
    if refiner is not None:
        print(f"\n{'='*60}")
        print(f"PIPELINE: {detector.backend.upper()} → "
            f"{ext_name.upper()} → "
            f"{tracker_name.upper()} → "
            f"GTALink({ref_name})")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"PIPELINE: {detector.backend.upper()} → "
            f"{ext_name.upper()} → "
            f"{tracker_name.upper()}")
        print(f"{'='*60}")
        
    if input_tracks is None:
        print("===== Chưa có kết quả MOT - Chạy toàn bộ pipeline ======")
        # Bước 1-2: MOT không chạy refiner, có trích xuất đặc trưng vào all_tracks nếu refiner = True
        all_tracks = run_mot(
            video_path      = video_path,
            detector        = detector,
            tracker         = tracker,
            extractor       = extractor,
            refiner         = refiner,
            extractor_refine= refiner_extractor,
        )
        # Bước 3: Offline refinement
        if refiner is not None:
            print(f"[OFFLINE TRACKER] Đang refine output bằng GTALink với extractor là {ref_name}")
            all_tracks = refine(all_tracks, refiner)
    else:
        print("===== Đã có kết quả MOT - Tái sử dụng ======")
        # Bước 3: Offline refinement
        if refiner is not None:
            print(f"[OFFLINE TRACKER] Đang refine output bằng GTALink với extractor là {ref_name}")
            all_tracks = refine(input_tracks, refiner)   #all_tracks có feat nhưng bỏ feat sau output refiner
        else:
            all_tracks = input_tracks   # Nếu không dùng refiner dùng trực tiếp input track để tính các bước tiếp theo


    if visualize:
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