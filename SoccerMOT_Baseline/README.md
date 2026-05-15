# SoccerMOT Baseline

**Pipeline:** YOLOv5 detect → OSNet feature → Pitch localization → Tracker (ByteTrack / StrongSORT / OC-SORT)

---

## Cấu trúc folder

```
SoccerMOT_Baseline/
├── run_baseline.py              ← File chạy chính
├── src/
│   ├── __init__.py
│   ├── pitch_localization.py   ← Bước 1: ánh xạ box → [x,y] pitch
│   ├── detector.py             ← YOLOv5 wrapper (tải pretrained tự động)
│   ├── feature_extractor.py    ← OSNet wrapper  (tải pretrained tự động)
│   ├── tracker.py              ← Wrapper thống nhất cho tất cả tracker
│   ├── osnet.py                ← Copy từ PipelineMOT/src/osnet.py
│   └── tracker_alg/
│       ├── __init__.py
│       ├── basetrack.py
│       ├── kalman_filter.py    ← KalmanFilter (8D) + KalmanFilterXY (4D)
│       ├── bytetrack.py        ← ByteTrack + use_project
│       ├── strongsort.py       ← StrongSORT + use_project
│       └── ocsort.py           ← OC-SORT + use_project
├── models/                     ← Đặt best_yolo26.pt vào đây (hoặc dùng pretrained)
└── output/                     ← Video output tự tạo
```

---

## Setup

### Bước 1: Copy osnet.py từ PipelineMOT
```bash
copy PipelineMOT\src\osnet.py  SoccerMOT_Baseline\src\osnet.py
```

### Bước 2: Chỉnh đường dẫn trong run_baseline.py
Mở `run_baseline.py`, chỉnh 3 biến đầu file:
```python
BASE       = r"D:\UITs subject\..."   # thư mục gốc project
YOLO_PATH  = os.path.join(BASE, r"PipelineMOT\models\best_yolo26.pt")
VIDEO_PATH = os.path.join(BASE, r"Data\wide_view\videos\F_20220220_1_1800_1830.mp4")
GT_CSV     = os.path.join(BASE, r"Data\wide_view\annotations\F_20220220_1_1800_1830.csv")
```

### Bước 3: Install dependencies
```bash
pip install torch torchvision opencv-python lap gdown ultralytics scipy
```

---

## Cách chạy

### Chạy mặc định (StrongSORT + use_project=True)
```bash
cd SoccerMOT_Baseline
python run_baseline.py
```

### Các tùy chọn
```bash
# ByteTrack, không dùng pitch projection
python run_baseline.py --tracker bytetrack

# StrongSORT với pitch projection
python run_baseline.py --tracker strongsort --use_project

# OC-SORT với pitch projection, chỉnh threshold
python run_baseline.py --tracker ocsort --use_project --distance_threshold 0.1

# Dùng pretrained YOLOv5s thay vì file custom
python run_baseline.py --yolo yolov5s
```

---

## Giải thích các bước pipeline

```
Bước 1: Pitch Localization
  box [x1,y1,x2,y2] → bottom-center (bx, by) → normalize → [x,y] ∈ [0,1]²
  File: src/pitch_localization.py

Bước 2: Kalman Filter cải tiến
  State gốc : [cx, cy, a, h, vx, vy, va, vh]  (8D, bị nhiễu bởi aspect ratio)
  State mới : [x,  y,  dx, dy]                 (4D, tọa độ sân, ý nghĩa vật lý)
  File: src/tracker_alg/kalman_filter.py → KalmanFilterXY

Bước 3: Load pretrained
  YOLOv5  : torch.hub hoặc ultralytics (tự tải nếu không có file local)
  OSNet   : gdown từ Google Drive (tự tải vào ~/.cache/torch/checkpoints/)

Bước 4: Pipeline chính
  detect (YOLOv5) → extract (OSNet) → pitch_xy (box_to_pitch_xy) → track
```

---

## So sánh use_project True vs False

| | `use_project=False` | `use_project=True` |
|---|---|---|
| Kalman state | `[cx,cy,a,h,vx,vy,va,vh]` 8D | `[x,y,dx,dy]` 4D |
| Distance | IoU | Euclidean (normalize) |
| Mahalanobis gating | chi²(4) = 9.49 | chi²(2) = 5.99 |
| Homography | Không | Tùy chọn (dùng normalize nếu chưa có) |

---

## Điều chỉnh distance_threshold

Khi `use_project=True` và **chưa có homography** (coordinate [0,1]²):

| Giá trị | Tương đương pixel (frame 1920px) |
|---|---|
| `0.05` | ~96px |
| `0.10` | ~192px ← thử trước |
| `0.15` | ~288px (default) |
| `0.20` | ~384px |

Khi có **homography** (coordinate tính bằng meter):
```python
H = np.load("homography.npy")
# truyền vào run_baseline():
run_baseline(..., H=H, tracker_kwargs={"distance_threshold": 3.0})
```

---

## Output format

`all_tracks` cùng format với PipelineMOT gốc:
```python
{
    track_id: {
        'boxes' : [None, [x1,y1,x2,y2], None, ...],  # None = frame không có box
        'frames': [frame_idx, ...]
    }
}
```
→ Dùng thẳng `compute_metrics()` từ PipelineMOT để so sánh.
