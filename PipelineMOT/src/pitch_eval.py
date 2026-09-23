import os
import re
import glob
import json
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Sequence

from src.compute_metrics import load_gt_from_csv
from src.pitch_localization import box_to_pitch_xy

FRAMES_PER_CLIP = 750
CLIP_REAL_FPS   = 30.0          # 750 frame ↔ 25 s thời gian thực
CLIP_SECONDS    = 30            # tên file <start>_<end> theo bước 30 s
ALPHAS          = np.arange(0.05, 0.99, 0.05)
_EPS            = np.finfo(float).eps


# GNSS ground truth

def _lonlat_to_local_m(lon, lat, lon0, lat0):
    """lon/lat (độ) → mét cục bộ (East, North), float64 (tránh lượng tử float32 ~1 m)."""
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return np.stack([(lon - lon0) * 111320.0 * np.cos(np.radians(lat0)),
                     (lat - lat0) * 110574.0], axis=-1)


def _apply_h(H, xy):
    xy1 = np.c_[np.asarray(xy, dtype=np.float64).reshape(-1, 2), np.ones(len(xy))] @ H.T
    return xy1[:, :2] / xy1[:, 2:3]


class GNSSGroundTruth:
    """Nối toàn bộ file GNSS thành một chuỗi thời gian, chiếu sang tọa độ sân (m)."""

    def __init__(self, gnss_dir: str, gnss_keypoints_path: str):
        with open(gnss_keypoints_path, "r") as f:
            raw = json.load(f)                                   # {"(x, y)": [lon, lat]}
        pitch = np.array([eval(k) for k in raw], dtype=np.float64)
        ll    = np.array(list(raw.values()), dtype=np.float64)
        self.lon0, self.lat0 = ll[:, 0].mean(), ll[:, 1].mean()
        H, _ = cv2.findHomography(_lonlat_to_local_m(ll[:, 0], ll[:, 1], self.lon0, self.lat0), pitch, 0)

        T, XY, uids = [], [], None
        for fp in sorted(glob.glob(os.path.join(gnss_dir, "G_*.csv"))):
            g = pd.read_csv(fp, header=None, dtype=str)
            cur = [int(float(tm)) * 100 + int(float(pl)) for tm, pl in zip(g.iloc[0, 1::2], g.iloc[1, 1::2])]
            if uids is None:
                uids = cur
            elif cur != uids:
                raise ValueError(f"Thứ tự cột GNSS khác nhau ở {fp}")
            ts  = pd.to_timedelta(g.iloc[4:, 0].str.strip()).dt.total_seconds().values
            lat = g.iloc[4:, 1::2].apply(pd.to_numeric, errors="coerce").values
            lon = g.iloc[4:, 2::2].apply(pd.to_numeric, errors="coerce").values
            m   = _lonlat_to_local_m(lon.ravel(), lat.ravel(), self.lon0, self.lat0)
            T.append(ts)
            XY.append(_apply_h(H, m).reshape(len(ts), -1, 2))
        if uids is None:
            raise FileNotFoundError(f"Không có file GNSS trong {gnss_dir}")

        T, idx    = np.unique(np.concatenate(T), return_index=True)   # bỏ timestamp trùng
        self.t    = T - (T[0] - 0.1)
        self.xy   = np.concatenate(XY)[idx]
        self.uids = uids
        self.col  = {u: j for j, u in enumerate(uids)}

    def positions_at(self, t: float) -> np.ndarray:
        """(P, 2) vị trí sân tại thời điểm t (nội suy tuyến tính); NaN nếu ngoài vùng phủ."""
        if t < self.t[0] or t > self.t[-1]:
            return np.full((len(self.uids), 2), np.nan)
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        w = (t - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        return self.xy[i - 1] * (1.0 - w) + self.xy[i] * w


def clip_index(video_name: str) -> int:
    """'F_20220220_1_0930_0960.mp4' → 31."""
    m = re.search(r"_(\d{4})_(\d{4})", os.path.basename(video_name))
    if m is None:
        raise ValueError(f"Không đọc được start time từ tên: {video_name}")
    return int(m.group(1)) // CLIP_SECONDS


def frame_time(k: int, frame_id: int, offset: float) -> float:
    """frame_id (1-based) của video k → thời điểm trên luồng GNSS (s)."""
    return (FRAMES_PER_CLIP * k + frame_id - 1) / CLIP_REAL_FPS + offset


# Bbox → sân

def _foot_xy(boxes_xyxy, H) -> np.ndarray:
    """[[x1,y1,x2,y2], ...] → (n,2) tọa độ sân, dùng đúng box_to_pitch_xy của pipeline."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 2))
    return np.array([box_to_pitch_xy([b[0], b[1], b[2] - b[0], b[3] - b[1]], H=H) for b in boxes_xyxy])


def _tracks_per_frame(all_tracks: Dict) -> Dict[int, List]:
    """all_tracks → {frame_id (1-based): [[x1,y1,x2,y2,tid], ...]}."""
    out = {}
    for tid, data in all_tracks.items():
        for fi, box in enumerate(data.get("boxes", [])):
            if box is not None:
                out.setdefault(fi + 1, []).append([*box[:4], tid])
    return out


def gt_as_tracks(gt_csv_path: str) -> Dict:
    """Bbox GT → định dạng all_tracks (dòng Oracle)."""
    gt = load_gt_from_csv(gt_csv_path)
    n, tracks = max(gt) if gt else 0, {}
    for f, boxes in gt.items():
        for x1, y1, x2, y2, uid in boxes:
            d = tracks.setdefault(int(uid), {"boxes": [None] * n, "frames": []})
            d["boxes"][f - 1] = [x1, y1, x2, y2]
            d["frames"].append(f - 1)
    return tracks


# Đồng bộ thời gian + khớp ID (chỉ dùng bbox GT)

def calibrate_clip(gnss: GNSSGroundTruth, gt: Dict, H: np.ndarray, k: int,
                   offsets: Sequence[float] = np.arange(-1.0, 4.0001, 0.05),
                   stride: int = 10) -> Dict:
    """Ước lượng offset δ (s) và ánh xạ bbox-ID → GNSS-ID cho video k."""
    frames = sorted(gt)[::stride]
    feet = {f: _foot_xy([b[:4] for b in gt[f]], H) for f in frames}
    ids  = {f: [int(b[4]) for b in gt[f]] for f in frames}

    def setmatch_err(off):                       # không cần ID
        e = []
        for f in frames:
            pg = gnss.positions_at(frame_time(k, f, off))
            pg = pg[~np.isnan(pg[:, 0])]
            if len(pg):
                D = np.linalg.norm(feet[f][:, None] - pg[None], axis=2)
                r, c = linear_sum_assignment(D)
                e.extend(D[r, c])
        return np.mean(e) if e else np.inf

    off0 = min(offsets, key=setmatch_err)
    bids = sorted({i for f in frames for i in ids[f]})
    S = np.zeros((len(bids), len(gnss.uids)))
    N = np.zeros_like(S)
    for f in frames:
        pg = gnss.positions_at(frame_time(k, f, off0))
        for p, i in zip(feet[f], ids[f]):
            d = np.linalg.norm(pg - p, axis=1)
            ok = ~np.isnan(d)
            S[bids.index(i), ok] += d[ok]
            N[bids.index(i), ok] += 1
    r, c = linear_sum_assignment(np.where(N > 0, S / np.maximum(N, 1), 1e6))
    id_map = {bids[i]: gnss.uids[j] for i, j in zip(r, c)}

    def id_err(off):                             # tinh chỉnh δ với ID đã khớp
        e = []
        for f in frames:
            pg = gnss.positions_at(frame_time(k, f, off))
            for p, i in zip(feet[f], ids[f]):
                d = np.linalg.norm(pg[gnss.col[id_map[i]]] - p)
                if not np.isnan(d):
                    e.append(d)
        return np.mean(e) if e else np.inf

    off = min(offsets, key=id_err)
    return {"offset": float(off), "id_map": id_map, "oracle_err": float(id_err(off))}


# Độ đo

def _pitch_hota(frames, n_gt, n_tr, D):
    """HOTA với độ tương đồng khoảng cách; matching 2 pha như TrackEval."""
    pot, gt_cnt, tr_cnt, sims = np.zeros((n_gt, n_tr)), np.zeros((n_gt, 1)), np.zeros((1, n_tr)), []
    for gi, gxy, ti, txy in frames:
        gt_cnt[gi, 0] += 1
        tr_cnt[0, ti] += 1
        s = None
        if len(gi) and len(ti):
            s = np.maximum(0.0, 1.0 - np.linalg.norm(gxy[:, None] - txy[None], axis=2) / D)
            den = s.sum(0)[None, :] + s.sum(1)[:, None] - s
            pot[np.ix_(gi, ti)] += np.where(den > _EPS, s / np.maximum(den, _EPS), 0)
        sims.append(s)
    gas = pot / np.maximum(gt_cnt + tr_cnt - pot, _EPS)

    A = len(ALPHAS)
    TP, FN, FP = np.zeros(A), np.zeros(A), np.zeros(A)
    mc = [np.zeros((n_gt, n_tr)) for _ in range(A)]
    for (gi, _, ti, _), s in zip(frames, sims):
        if s is None:
            FN += len(gi); FP += len(ti)
            continue
        r, c = linear_sum_assignment(-(gas[np.ix_(gi, ti)] * s))
        for a, alpha in enumerate(ALPHAS):
            ok = s[r, c] >= alpha - _EPS
            n = int(ok.sum())
            TP[a] += n; FN[a] += len(gi) - n; FP[a] += len(ti) - n
            mc[a][np.asarray(gi)[r[ok]], np.asarray(ti)[c[ok]]] += 1

    det_a = TP / np.maximum(TP + FN + FP, 1)
    ass_a = np.array([(m * m / np.maximum(gt_cnt + tr_cnt - m, 1)).sum() / max(TP[a], 1)
                      for a, m in enumerate(mc)])
    return 100 * float(np.sqrt(det_a * ass_a).mean())


def _gospa_cost(ng, nt, D, c):
    """Chi phí GOSPA tối ưu 1 frame (p=1, α=2): d nếu ghép & d < c; c/2 mỗi phần tử lẻ."""
    C = np.full((ng + nt, nt + ng), 1e9)
    if ng and nt:
        C[:ng, :nt] = np.where(D < c, D, 1e9)
    C[np.arange(ng), nt + np.arange(ng)] = c / 2
    C[ng + np.arange(nt), np.arange(nt)] = c / 2
    C[ng:, nt:] = 0.0
    r, cc = linear_sum_assignment(C)
    return float(C[r, cc].sum())


def _errors(frames, n_gt, n_tr, c):
    """(E_frame, E_1to1) — mét / cầu thủ / frame."""
    n_slots = max(sum(len(f[0]) for f in frames), 1)
    e_frame, B = 0.0, np.zeros((n_tr, n_gt))
    for gi, gxy, ti, txy in frames:
        D = np.linalg.norm(gxy[:, None] - txy[None], axis=2) if len(gi) and len(ti) else np.zeros((len(gi), len(ti)))
        e_frame += _gospa_cost(len(gi), len(ti), D, c)
        if D.size:
            B[np.ix_(ti, gi)] += np.maximum(0.0, c - D.T)       # lợi ích gán track → cầu thủ

    r, g = linear_sum_assignment(-B)                            # mỗi cầu thủ ≤ 1 track
    assign = {int(t): int(p) for t, p in zip(r, g) if B[t, p] > 0}
    e_1to1 = 0.0
    for gi, gxy, ti, txy in frames:
        pos, covered = {p: q for q, p in enumerate(gi)}, set()
        for b, t in enumerate(ti):
            p = assign.get(t)
            d = np.linalg.norm(txy[b] - gxy[pos[p]]) if p in pos else np.inf
            if d < c:
                e_1to1 += d; covered.add(p)
            else:
                e_1to1 += c / 2                                 # track thừa / sai người
        e_1to1 += c / 2 * (len(gi) - len(covered))              # cầu thủ bị bỏ sót
    return e_frame / n_slots, e_1to1 / n_slots


def evaluate_clip(all_tracks: Dict, gt_csv_path: str, video_name: str,
                  gnss: GNSSGroundTruth, H: np.ndarray, calib: Optional[Dict] = None,
                  D: float = 10.0, c: float = 10.0) -> Dict:
    """Pitch-HOTA, E_1to1 (và E_frame) cho một video."""
    k  = clip_index(video_name)
    gt = load_gt_from_csv(gt_csv_path)
    calib = calib or calibrate_clip(gnss, gt, H, k)
    pred  = _tracks_per_frame(all_tracks)
    tcol  = {t: i for i, t in enumerate(sorted(all_tracks))}

    frames = []
    for f in sorted(gt):
        pg = gnss.positions_at(frame_time(k, f, calib["offset"]))
        ok = ~np.isnan(pg[:, 0])
        if ok.any():
            pb = pred.get(f, [])
            frames.append((np.where(ok)[0].tolist(), pg[ok],
                           [tcol[b[4]] for b in pb], _foot_xy([b[:4] for b in pb], H)))

    n_tr = max(len(tcol), 1)
    e_frame, e_1to1 = _errors(frames, len(gnss.uids), n_tr, c)
    return {"video": os.path.basename(video_name),
            "Pitch-HOTA": _pitch_hota(frames, len(gnss.uids), n_tr, D),
            "E_1to1": e_1to1, "E_frame": e_frame}


def summarize(results: List[Dict]) -> Dict:
    """Trung bình theo video."""
    return {k: float(np.mean([r[k] for r in results])) for k in ["Pitch-HOTA", "E_1to1", "E_frame"]}
