"""
save_pitch_tracks.py — chạy E2ESoTrack trên 6 clip có GNSS (720–900 s) và lưu tracks cho eval_pitch.py.

    Output/pitch_eval/tracks_e2e_fast.pkl  : YOLO26 → ByteTrack
    Output/pitch_eval/tracks_e2e_best.pkl  : YOLO26 → ByteTrack → GTALink(SOLIDER)

Mỗi file là {video_name: {track_id: {'boxes', 'frames'}}}, giống các cell 3a/3b trong test.ipynb.
Tracks của baseline (tracks_baseline*.pkl) được tạo từ SoccerMOT_Baseline/main.ipynb.

    python save_pitch_tracks.py [--out ../Output/pitch_eval]
"""

import argparse
import os
import pickle

import demo

VIDEOS_DIR = os.path.join(demo.DATA, "wide_view", "videos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(demo.BASE, "Output", "pitch_eval"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    videos = sorted(os.listdir(VIDEOS_DIR))[24:30]          # F_..._0720_0750 → F_..._0870_0900
    detector, extractor_refine = demo.build_components("optimal")

    fast, best = {}, {}
    for v in videos:
        print(f"\n=== {v} ===")
        online, _, _ = demo.track_online(os.path.join(VIDEOS_DIR, v), detector, extractor_refine)
        fast[v] = demo.strip_feats(online)
        best[v] = demo.strip_feats(demo.refine_offline(online))

    for name, tracks in [("tracks_e2e_fast.pkl", fast), ("tracks_e2e_best.pkl", best)]:
        with open(os.path.join(args.out, name), "wb") as f:
            pickle.dump(tracks, f)
        print("saved", os.path.join(args.out, name))


if __name__ == "__main__":
    main()
