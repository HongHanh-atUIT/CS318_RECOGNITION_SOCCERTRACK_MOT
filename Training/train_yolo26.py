import os
import glob
from ultralytics import YOLO


# CONFIG
WORKSPACE   = "/workspace"
DATA_YAML   = os.path.join(WORKSPACE, "soccer.yaml")
MODEL_PT    = os.path.join(WORKSPACE, "yolo26m.pt")
EPOCHS      = 50
IMGSZ       = 1600
BATCH       = 8
DEVICE      = "cuda"


# TẠO SOCCER.YAML
yaml_content = f"""
train: {WORKSPACE}/detection/images/train
val:   {WORKSPACE}/detection/images/val
test:  {WORKSPACE}/detection/images/test
nc: 1
names: ['player']
""".strip()

with open(DATA_YAML, "w") as f:
    f.write(yaml_content)
print(f"Đã tạo {DATA_YAML}")
print(open(DATA_YAML).read())


# TRAIN
def get_latest_run(base_dir=os.path.join(WORKSPACE, "runs/detect")):
    runs = sorted(glob.glob(os.path.join(base_dir, "train*")),
                  key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError(f"Không tìm thấy thư mục train nào trong {base_dir}")
    return runs[-1]

print("\n========== BẮT ĐẦU TRAIN ==========")
model = YOLO(MODEL_PT)
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    rect=True,
    plots=True,
    workers=4,
    project=os.path.join(WORKSPACE, "runs/detect"),
    name="train"
)


# ĐÁNH GIÁ TRÊN TEST SET
latest_run = get_latest_run()
best_pt    = os.path.join(latest_run, "weights", "best.pt")
print(f"\n========== ĐÁNH GIÁ ==========")
print(f"Dùng model: {best_pt}")

model   = YOLO(best_pt)
metrics = model.val(data=DATA_YAML, split="test")

print("\n========== KẾT QUẢ TEST SET ==========")
print(f"Precision:  {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall:     {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50:      {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95:   {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")


# PREDICT MẪU
test_img = os.path.join(WORKSPACE, "detection/images/test")

# Lấy 1 ảnh bất kỳ trong test set
test_images = glob.glob(os.path.join(test_img, "*.jpg"))
if test_images:
    sample_img = test_images[0]
    print(f"\n========== PREDICT MẪU ==========")
    print(f"Ảnh: {sample_img}")
    model.predict(
        source=sample_img,
        save=True,
        conf=0.25,
        show_labels=False,
        show_conf=False,
        line_width=1,
        project=os.path.join(WORKSPACE, "runs/detect"),
        name="predict"
    )
    print(f"Kết quả lưu tại: {WORKSPACE}/runs/detect/predict/")
else:
    print("Không tìm thấy ảnh test!")