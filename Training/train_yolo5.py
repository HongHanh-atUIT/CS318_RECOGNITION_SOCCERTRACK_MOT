import os
import glob
from ultralytics import YOLO

# CONFIG
WORKSPACE   = "/workspace"
DATA_YAML   = os.path.join(WORKSPACE, "soccer.yaml")
MODEL_PT    = os.path.join(WORKSPACE, "yolov5m.pt")  
EPOCHS      = 50
IMGSZ       = 1600
BATCH       = 8
DEVICE      = "cuda"

# TẠO SOCCER.YAML
yaml_content = f"""train: {WORKSPACE}/detection/images/train
val:   {WORKSPACE}/detection/images/val
test:  {WORKSPACE}/detection/images/test
nc: 1
names: ['player']""".strip()

with open(DATA_YAML, "w") as f:
    f.write(yaml_content)
print(open(DATA_YAML).read())

# TRAIN
def get_latest_run(base_dir=os.path.join(WORKSPACE, "runs/detect")):
    runs = sorted(glob.glob(os.path.join(base_dir, "train*")),
                  key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError(f"Không tìm thấy run nào trong {base_dir}")
    return runs[-1]

print("\n========== TRAIN YOLOv5m ==========")
model = YOLO(MODEL_PT)
model.train(
    data    = DATA_YAML,
    epochs  = EPOCHS,
    imgsz   = IMGSZ,
    batch   = BATCH,
    device  = DEVICE,
    rect    = True,
    plots   = True,
    workers = 4,
    project = os.path.join(WORKSPACE, "runs/detect"),
    name    = "train_yolov5",  
)

# ĐÁNH GIÁ
latest_run = get_latest_run()
best_pt    = os.path.join(latest_run, "weights", "best.pt")
print(f"\nBest model: {best_pt}")

model   = YOLO(best_pt)
metrics = model.val(data=DATA_YAML, split="test")

print("\n========== KẾT QUẢ TEST SET ==========")
print(f"Precision : {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall    : {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50     : {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95  : {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

# PREDICT MẪU
test_images = glob.glob(os.path.join(WORKSPACE, "detection/images/test/*.jpg"))
if test_images:
    model.predict(
        source     = test_images[0],
        save       = True,
        conf       = 0.25,
        show_labels= False,
        show_conf  = False,
        line_width = 1,
        project    = os.path.join(WORKSPACE, "runs/detect"),
        name       = "predict_yolov5",
    )