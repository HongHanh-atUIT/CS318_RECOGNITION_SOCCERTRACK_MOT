import torch.cuda
torch.cuda.is_bf16_supported = lambda *args, **kwargs: False

from tqdm import tqdm
import tqdm as tqdm_module
tqdm_module.tqdm.__init__.__defaults__ = tuple(
    True if i == 7 else v
    for i, v in enumerate(tqdm_module.tqdm.__init__.__defaults__)
)

import os
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt

from rfdetr import RFDETRMedium


def main():
    # ============================================================
    # CONFIG
    # ============================================================
    BASE_DIR       = "/workspace"
    DATASET_DIR    = os.path.join(BASE_DIR, "RF_Dataset")
    RESULT_DIR     = os.path.join(BASE_DIR, "RF_DETR_result")
    CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
    LOG_DIR        = os.path.join(RESULT_DIR, "logs")
    PLOT_DIR       = os.path.join(RESULT_DIR, "plots")

    for d in [RESULT_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"Kết quả lưu tại: {RESULT_DIR}")

    # ============================================================
    # KIỂM TRA DATASET
    # ============================================================
    print("\n[DEBUG] Kiểm tra dataset:")
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(DATASET_DIR, split)
        n_imgs    = len(glob.glob(os.path.join(split_dir, "*.jpg")))
        has_json  = os.path.exists(os.path.join(split_dir, "_annotations.coco.json"))
        print(f"  {split}: {n_imgs} ảnh | JSON: {has_json}")

    # ============================================================
    # TRAIN
    # ============================================================
    print("\n========== BẮT ĐẦU TRAIN ==========")

    # num_classes=1 vì chỉ có class 'player'
    # resolution=576 là chuẩn của RFDETRMedium
    model = RFDETRMedium(num_classes=1)

    model.train(
        dataset_dir=DATASET_DIR,
        epochs=50,
        batch_size=4,
        grad_accum_steps=4,      # effective batch = 16
        lr=5e-5,
        resolution=576,          # resolution chuẩn của Medium
        output_dir=CHECKPOINT_DIR,
        device="cuda",
        tensorboard=False,
        amp=True,                # mixed precision tiết kiệm VRAM
        early_stopping=True,
        early_stopping_patience=10,
        val_interval=1,
        num_workers=4,
    )

    print("Training hoàn tất!")

    # ============================================================
    # COPY BEST MODEL VỀ RESULT_DIR
    # ============================================================
    best_files = [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
    ]
    print("\n========== LƯU CHECKPOINT ==========")
    for file in best_files:
        src = os.path.join(CHECKPOINT_DIR, file)
        if os.path.exists(src):
            dst = os.path.join(RESULT_DIR, file)
            shutil.copy(src, dst)
            print(f"Đã copy: {file}")
        else:
            print(f"Không tìm thấy: {file}")

    # ============================================================
    # COPY METRICS
    # ============================================================
    metrics_src = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    if os.path.exists(metrics_src):
        shutil.copy(metrics_src, os.path.join(LOG_DIR, "metrics.csv"))
        print("Đã copy metrics.csv vào logs/")

    # ============================================================
    # PLOT
    # ============================================================
    metrics_path = os.path.join(LOG_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        print(f"\nCác cột trong metrics.csv: {df.columns.tolist()}")

        plt.figure(figsize=(12, 8))
        for col, label in [
            ("train/loss",    "Train Loss"),
            ("val/mAP_50_95", "Val mAP@50-95"),
            ("val/mAP_50",    "Val mAP@50"),
        ]:
            if col in df.columns:
                plt.plot(df["epoch"], df[col], label=label)

        plt.title("RF-DETR Medium - Training Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(PLOT_DIR, "training_metrics.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Đã lưu biểu đồ: {plot_path}")

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    print("\n" + "=" * 60)
    print("HOÀN TẤT! Các file cần tải về máy:")
    for file in best_files:
        p = os.path.join(RESULT_DIR, file)
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024 / 1024
            print(f"  {p}  ({size_mb:.1f} MB)")
    print(f"  {os.path.join(LOG_DIR, 'metrics.csv')}")
    print(f"  {os.path.join(PLOT_DIR, 'training_metrics.png')}")


if __name__ == "__main__":
    main()