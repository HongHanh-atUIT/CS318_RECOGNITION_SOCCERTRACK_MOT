import os
import sys
import pickle
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.pitch_localization import load_homography
from src.compute_metrics import load_gt_from_csv
from src.pitch_eval import (GNSSGroundTruth, calibrate_clip, clip_index, gt_as_tracks,
                            evaluate_clip, summarize)

BASE       = os.path.dirname(HERE)
DATA       = os.path.join(BASE, "Data")
VIDEOS_DIR = os.path.join(DATA, "wide_view", "videos")
GT_DIR     = os.path.join(DATA, "wide_view", "annotations")
PITCH_DIR  = os.path.join(BASE, "Output", "pitch_eval")

METHODS = {                                   # tên hiển thị → file track (None = Oracle)
    "Oracle (GT boxes)"        : None,
    "Baseline"                 : "tracks_baseline.pkl",
    "Baseline (trained)"       : "tracks_baseline_trained.pkl",
    "YOLO26 + ByteTrack"       : "tracks_e2e_fast.pkl",
    "YOLO26 + ByteTrack + GTA" : "tracks_e2e_best.pkl",
}


def main():
    videos = sorted(os.listdir(VIDEOS_DIR))[24:30]
    gt_csv = lambda v: os.path.join(GT_DIR, os.path.splitext(v)[0] + ".csv")

    gnss = GNSSGroundTruth(os.path.join(DATA, "GNSS"), os.path.join(DATA, "gnss_keypoints.json"))
    H, _ = load_homography(os.path.join(DATA, "fisheye_keypoints.json"))
    calibs = {v: calibrate_clip(gnss, load_gt_from_csv(gt_csv(v)), H, clip_index(v)) for v in videos}
    for v, cal in calibs.items():
        print(f"{v}: offset {cal['offset']:+.2f} s")

    rows, per_video = [], []
    for name, fname in METHODS.items():
        if fname is None:
            tracks = {v: gt_as_tracks(gt_csv(v)) for v in videos}
        else:
            with open(os.path.join(PITCH_DIR, fname), "rb") as f:
                tracks = pickle.load(f)
        res = [evaluate_clip(tracks[v], gt_csv(v), v, gnss, H, calib=calibs[v]) for v in videos]
        rows.append({"Pipeline": name, **summarize(res)})
        per_video += [{"Pipeline": name, **r} for r in res]

    table = pd.DataFrame(rows).set_index("Pipeline").round(2)
    print("\n", table[["Pitch-HOTA", "E_1to1"]].to_string(), sep="")
    print("\n(E_frame — chỉ để phân tích)\n", table["E_frame"].to_string(), sep="")

    os.makedirs(PITCH_DIR, exist_ok=True)
    table.to_csv(os.path.join(PITCH_DIR, "pitch_eval.csv"))
    pd.DataFrame(per_video).round(3).to_csv(os.path.join(PITCH_DIR, "pitch_eval_per_video.csv"), index=False)
    print("\n" + table[["Pitch-HOTA", "E_1to1"]].to_latex(float_format="%.2f"))


if __name__ == "__main__":
    main()
