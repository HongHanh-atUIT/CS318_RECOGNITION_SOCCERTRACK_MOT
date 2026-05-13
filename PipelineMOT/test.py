import cv2
import os
import numpy as np
from typing import List, Dict, Optional
from src.feature_extractor import FeatureExtractor
from src.detector import Detector
from src.tracker import Tracker
from src.gtalink import GTALink
from src.compute_metrics import compute_metrics
from src.utils import *

# ============================================================
# Path & Config
# ============================================================

# BASE = r"D:\UITs subject\Năm 3\Nhận dạng\CS318_RECOGNITION_SOCCERTRACK_MOT"
BASE = r"D:\UIT 3rd year\NhanDang\Project"

# models
YOLO_PATH    = os.path.join(BASE, r"PipelineMOT\models\best_yolo26.pt")
RF_PATH      = os.path.join(BASE, r"PipelineMOT\models\best_rf_detr.pth")
SOLIDER_PATH = os.path.join(BASE, r"PipelineMOT\models\swin_small_converted.pth")

# data
VIDEOS_DIR      = r"D:\UIT 3rd year\NhanDang\Project\Data\wide_view\videos"
GROUNDTRUTH_DIR = r"D:\UIT 3rd year\NhanDang\Project\Data\wide_view\annotations"
ALL_VIDEOS      = sorted(os.listdir(VIDEOS_DIR))
ALL_VIDEOS      = ALL_VIDEOS[-6:]

# output
OUTPUT_DIR = os.path.join(BASE, r"Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# all config
ALL_DETECTOR  = ["yolo26", "rf_detr"]
ALL_EXTRACTOR = ["osnet", "solider", "color_histogram"]
ALL_TRACKER   = ["bytetrack", "ocsort", "strongsort", "deepeiou"]


# ============================================================
# Testing function
# ============================================================

def Testing(videos, detector, extractor,
            tracker, refiner, extractor_refiner,
            tracker_kwargs: Optional[Dict] = None,
            visualize=False, output_video=None,
            input_tracks_map=None,  # dict {vid_name: all_tracks} để tái sử dụng
            **viz_kwargs):

    all_results = {}
    metrics_accum = {}
    all_tracks_map = {}  # dict lưu all_tracks mỗi video để tái sử dụng

    # Xác định tên các thành phần cho tên file
    det_name     = detector.backend if detector is not None else "None"
    ext_name     = extractor.backend if extractor is not None else "None"
    trk_name     = tracker if tracker is not None else "None"
    ref_ext_name = (extractor_refiner.backend if extractor_refiner is not None else
                    extractor.backend if extractor is not None else "None") \
                   if refiner is not None else "None"
                   
    if refiner is not None:
        print(f"\n{'='*60}")
        print(f"PIPELINE: {det_name.upper()} → "
            f"{ext_name.upper()} → "
            f"{trk_name.upper()} → "
            f"GTALink({ref_ext_name})")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"PIPELINE: {det_name.upper()} → "
            f"{ext_name.upper()} → "
            f"{trk_name.upper()}")
        print(f"{'='*60}")

    result_filename = f"result_{det_name}_{ext_name}_{trk_name}_{ref_ext_name}.txt"
    result_path = os.path.join(OUTPUT_DIR, result_filename)

    for vid in videos:
        video_path = os.path.join(VIDEOS_DIR, vid)
        gt_path    = os.path.join(GROUNDTRUTH_DIR, vid.split('.')[0] + ".csv")

        print(f"\n{'='*60}")
        print(f"Processing video: {vid}")
        print(f"{'='*60}")

        # Lấy cached tracks nếu có (tránh chạy lại MOT)
        input_tracks = input_tracks_map.get(vid) if input_tracks_map is not None else None

        result = run_pipeline(
            video_path        = video_path,
            gt_csv_path       = gt_path,
            detector          = detector,
            tracker_name      = tracker,
            output_path       = output_video,
            extractor         = extractor,
            refiner           = refiner,
            refiner_extractor = extractor_refiner,
            tracker_kwargs    = tracker_kwargs or {},
            visualize         = visualize,
            input_tracks      = input_tracks,
            **viz_kwargs,
        )

        metrics = result['metrics']
        all_results[vid] = metrics
        all_tracks_map[vid] = result['all_tracks']  # lưu all_tracks theo tên video

        # Tích luỹ metrics để tính trung bình
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_accum.setdefault(k, []).append(v)

    # Tính trung bình
    avg_metrics = {k: np.mean(vals) for k, vals in metrics_accum.items()}

    print(f"\n{'='*60}")
    print("AVERAGE METRICS ACROSS ALL VIDEOS:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")

    # Lưu kết quả ra file
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Pipeline: {det_name} -> {ext_name} -> {trk_name} -> GTALink({ref_ext_name})\n")
        f.write(f"{'='*60}\n\n")
        for vid, metrics in all_results.items():
            f.write(f"[{vid}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
            f.write("\n")
        f.write(f"{'='*60}\n")
        f.write("AVERAGE:\n")
        for k, v in avg_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")

    print(f"\nKết quả đã lưu tại: {result_path}")

    return {
        'per_video':      all_results,
        'average':        avg_metrics,
        'all_tracks_map': all_tracks_map,  # {vid_name: all_tracks}
    }


def main():
    # ============================================================
    # Khởi tạo models
    # ============================================================

    detector_yolo26   = Detector(backend="yolo26", model_path=YOLO_PATH)
    detector_rfdetr   = Detector(backend="rf_detr", model_path=RF_PATH)
    extractor_color   = FeatureExtractor(backend="color_histogram")
    extractor_osnet   = FeatureExtractor(backend="osnet")
    extractor_solider = FeatureExtractor(
        backend="solider",
        solider_model_path=SOLIDER_PATH,
        solider_arch="swin_small",
        solider_semantic_weight=0.2,
    )
    refiner = GTALink(merge_dist_thres=0.35)

    # ============================================================
    # YOLO26 - OSNET
    # ============================================================

    # yolo26 - osnet - deepeiou - None
    result_1 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "deepeiou", refiner=None, extractor_refiner=None,
                       input_tracks_map=None, tracker_kwargs=None)

    # yolo26 - osnet - deepeiou - solider
    result_2 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "deepeiou", refiner=refiner, extractor_refiner=extractor_solider,
                       input_tracks_map=result_1['all_tracks_map'],
                       tracker_kwargs=None)
    
    # yolo26 - osnet - ocsort - None
    result_1 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "ocsort", refiner=None, extractor_refiner=None,
                       input_tracks_map=None, 
                       tracker_kwargs={"use_project": False, "use_emb": True})

    # yolo26 - osnet - ocsort - solider
    result_2 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "ocsort", refiner=refiner, extractor_refiner=extractor_solider,
                       input_tracks_map=result_1['all_tracks_map'],
                       tracker_kwargs={"use_project": False, "use_emb": True})
    
    # yolo26 - osnet - strongsort - None
    result_1 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "strongsort", refiner=None, extractor_refiner=None,
                       input_tracks_map=None, 
                       tracker_kwargs=None)

    # yolo26 - osnet - strongsort - solider
    result_2 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "strongsort", refiner=refiner, extractor_refiner=extractor_solider,
                       input_tracks_map=result_1['all_tracks_map'],
                       tracker_kwargs=None)
    
    # yolo26 - osnet - bytetrack - None
    result_1 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "bytetrack", refiner=None, extractor_refiner=None,
                       input_tracks_map=None, 
                       tracker_kwargs={"use_project": False, "lambda_iou": 0.8})

    # yolo26 - osnet - bytetrack - solider
    result_2 = Testing(ALL_VIDEOS, detector_yolo26, extractor_osnet,
                       "bytetrack", refiner=refiner, extractor_refiner=extractor_solider,
                       input_tracks_map=result_1['all_tracks_map'],
                       tracker_kwargs={"use_project": False, "lambda_iou": 0.8})

    # ============================================================
    # RF-DETR
    # ============================================================


if __name__ == "__main__":
    main()