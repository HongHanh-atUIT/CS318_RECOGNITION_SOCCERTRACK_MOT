# SoccerMOT Baseline — YOLOv5 → OSNet → DeepSORT

Baseline used for comparison in the E2ESoTrack paper (Table 4 & Table 5).

| Variant | Detector weights | Backend |
|---|---|---|
| **Baseline** | `yolov5m.pt` (COCO pretrained) | `yolov5_hub` (local `yolov5/` repo via `torch.hub`) |
| **Baseline (trained)** | `best.pt` (YOLOv5 retrained on SoccerTrack) | `yolov5` (ultralytics) |

Both variants share the same OSNet-x1.0 appearance model and the same modified DeepSORT
(`TRACKER_KWARGS` in `main.ipynb`: `use_project=True, n_init=1, track_high_thresh=0.6,
new_track_thresh=0.65, max_age=60, max_cosine_distance=0.4, max_iou_distance=0.7`).

## Layout

```
SoccerMOT_Baseline/
├── main.ipynb              # run baseline on test set + GNSS clips, compute metrics
├── plot_keypoints.ipynb    # visualise homography keypoints on the wide-view frame
├── yolov5/                 # (not tracked) clone of ultralytics/yolov5 for torch.hub
└── src/
    ├── detector.py         # YOLOv5 wrapper (yolov5_hub | yolov5)
    ├── feature_extractor.py# OSNet wrapper (weights auto-downloaded with gdown)
    ├── osnet.py
    ├── tracker.py          # DeepSORT wrapper
    ├── tracker_alg/        # basetrack, kalman_filter (8D + 4D pitch), deepsort
    ├── pitch_localization.py
    ├── compute_metrics.py  # MOTA / MOTP / IDF1 / HOTA
    ├── utils.py            # run_pipeline(...)
    └── models/             # (not tracked) yolov5m.pt, best.pt
```

## Setup

```bash
git clone https://github.com/ultralytics/yolov5 SoccerMOT_Baseline/yolov5
pip install -r SoccerMOT_Baseline/yolov5/requirements.txt
```

Put `yolov5m.pt` and `best.pt` in `SoccerMOT_Baseline/src/models/`.

## Run

Open `main.ipynb` with `SoccerMOT_Baseline/` as working directory (imports use `src.*`), set `BASE`
to the repository root, then run the cells in order:

1. **Components** — detectors, OSNet, `TRACKER_KWARGS`.
2. **Pitch-localisation clips** — runs both baselines on the six GNSS clips (`ALL_VIDEOS[24:30]`)
   and saves `Output/pitch_eval/tracks_baseline*.pkl`, consumed by `PipelineMOT/eval_pitch.py`.
3. **Test set** — runs on `ALL_VIDEOS[-6:]` and averages MOTA / IDF1 / HOTA.

Output format of `run_pipeline(...)["all_tracks"]` is identical to `PipelineMOT`:

```python
{track_id: {"boxes": [None | [x1, y1, x2, y2], ...],   # one entry per frame
            "frames": [frame_idx, ...]}}
```
