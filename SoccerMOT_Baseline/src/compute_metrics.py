"""
compute_metrics.py
------------------
Tính các độ đo MOT chuẩn:
    - MOTA  (Multiple Object Tracking Accuracy)
    - MOTP  (Multiple Object Tracking Precision)
    - IDF1  (ID F1 Score)
    - HOTA  (Higher Order Tracking Accuracy)
    - IDs   (ID Switch count)

Cách dùng:
    from src.compute_metrics import compute_metrics

    metrics = compute_metrics(
        all_tracks   = all_tracks,       # output của run_mot / run_and_visualize
        csv_path     = "path/to/gt.csv", # file annotation CSV
        iou_thresh   = 0.5,
    )
    print(metrics)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# Đọc Ground Truth từ CSV
# ─────────────────────────────────────────────────────────────

def load_gt_from_csv(csv_path: str) -> Dict[int, List[List[float]]]:
    """
    Đọc file annotation CSV và trả về ground truth.

    Format CSV:
        Row 0: TeamID
        Row 1: PlayerID
        Row 2: Attributes (bb_height, bb_left, bb_top, bb_width lặp lại)
        Row 3: "frame" header
        Row 4+: dữ liệu theo từng frame

    Returns
    -------
    gt : dict
        {frame_id (1-based): [[x1, y1, x2, y2, gt_id], ...]}
        Chỉ bao gồm player (bỏ BALL)
    """
    raw = pd.read_csv(csv_path, header=None)

    # Dòng 0: TeamID, Dòng 1: PlayerID, Dòng 2: Attributes
    team_row   = raw.iloc[0].tolist()
    player_row = raw.iloc[1].tolist()
    attr_row   = raw.iloc[2].tolist()

    # Xây dựng danh sách object (bỏ BALL)
    # Mỗi object có 4 cột: bb_height, bb_left, bb_top, bb_width
    objects = []   # list of (gt_id, h_col, l_col, t_col, w_col)
    gt_id   = 0
    col     = 1
    while col + 3 < len(team_row):
        team   = str(team_row[col]).strip().upper()
        player = str(player_row[col]).strip()

        if team == "BALL":
            col += 4
            continue

        # Tạo gt_id duy nhất từ team + player
        unique_id = int(float(team)) * 100 + int(float(player))
        objects.append((unique_id, col, col + 1, col + 2, col + 3))
        col += 4

    # Duyệt từng frame (từ dòng 4 trở đi)
    data_rows = raw.iloc[4:]
    gt = {}

    for _, row in data_rows.iterrows():
        vals = row.tolist()
        try:
            frame_id = int(float(str(vals[0]).strip()))
        except (ValueError, TypeError):
            continue

        boxes = []
        for (uid, h_col, l_col, t_col, w_col) in objects:
            try:
                h    = float(vals[h_col])
                left = float(vals[l_col])
                top  = float(vals[t_col])
                w    = float(vals[w_col])
            except (ValueError, IndexError, TypeError):
                continue

            if any(np.isnan(v) for v in [h, left, top, w]):
                continue
            if w <= 0 or h <= 0:
                continue

            x1 = left
            y1 = top
            x2 = left + w
            y2 = top  + h
            boxes.append([x1, y1, x2, y2, uid])

        if boxes:
            gt[frame_id] = boxes

    return gt


# ─────────────────────────────────────────────────────────────
# Chuyển all_tracks sang format per-frame
# ─────────────────────────────────────────────────────────────

def all_tracks_to_per_frame(
    all_tracks: Dict
) -> Dict[int, List[List[float]]]:
    """
    Chuyển all_tracks (output run_mot) sang format per-frame.

    Returns
    -------
    pred : dict
        {frame_id (1-based): [[x1, y1, x2, y2, track_id], ...]}
    """
    pred = defaultdict(list)
    for tid, data in all_tracks.items():
        for fi, box in enumerate(data.get('boxes', [])):
            if box is None:
                continue
            frame_id = fi + 1   # 0-based → 1-based
            x1, y1, x2, y2 = box[:4]
            pred[frame_id].append([x1, y1, x2, y2, tid])
    return dict(pred)


# ─────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────

def _iou(box1: List[float], box2: List[float]) -> float:
    """Tính IoU giữa 2 box [x1,y1,x2,y2]."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-8
    return inter / union


def _match_detections(
    gt_boxes:   List[List[float]],
    pred_boxes: List[List[float]],
    iou_thresh: float
) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Greedy matching giữa GT và Pred dựa trên IoU.

    Returns
    -------
    matches      : list of (gt_idx, pred_idx)
    unmatched_gt : list of gt_idx không được match
    unmatched_pred: list of pred_idx không được match
    """
    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou_matrix[i, j] = _iou(g, p)

    matches       = []
    matched_gt    = set()
    matched_pred  = set()

    # Greedy: lấy cặp có IoU cao nhất trước
    flat_indices = np.argsort(-iou_matrix.flatten())
    for idx in flat_indices:
        i, j = divmod(idx, len(pred_boxes))
        if iou_matrix[i, j] < iou_thresh:
            continue
        if i not in matched_gt and j not in matched_pred:
            matches.append((i, j))
            matched_gt.add(i)
            matched_pred.add(j)

    unmatched_gt   = [i for i in range(len(gt_boxes))   if i not in matched_gt]
    unmatched_pred = [j for j in range(len(pred_boxes)) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


# ─────────────────────────────────────────────────────────────
# MOTA / MOTP
# ─────────────────────────────────────────────────────────────

def _compute_mota_motp(
    gt_per_frame:   Dict[int, List],
    pred_per_frame: Dict[int, List],
    iou_thresh:     float
) -> Dict:
    """
    Tính MOTA và MOTP theo chuẩn MOTChallenge.

    MOTA = 1 - (FP + FN + IDs) / GT_total
    MOTP = sum(IoU of matched) / total_matched
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_switch = 0
    total_iou = 0.0
    total_matched = 0

    # Theo dõi ID switch: gt_id → pred_id frame trước
    prev_gt_to_pred: Dict[int, int] = {}

    all_frames = sorted(set(gt_per_frame) | set(pred_per_frame))

    for frame_id in all_frames:
        gt_boxes   = gt_per_frame.get(frame_id, [])
        pred_boxes = pred_per_frame.get(frame_id, [])

        total_gt += len(gt_boxes)

        if not gt_boxes:
            total_fp += len(pred_boxes)
            continue

        matches, unmatched_gt, unmatched_pred = _match_detections(
            [g[:4] for g in gt_boxes],
            [p[:4] for p in pred_boxes],
            iou_thresh
        )

        total_fn += len(unmatched_gt)
        total_fp += len(unmatched_pred)

        curr_gt_to_pred = {}
        for gi, pi in matches:
            gt_id   = gt_boxes[gi][4]
            pred_id = pred_boxes[pi][4]
            iou_val = _iou(gt_boxes[gi][:4], pred_boxes[pi][:4])
            total_iou     += iou_val
            total_matched += 1

            curr_gt_to_pred[gt_id] = pred_id

            # ID switch: gt_id đã được match frame trước với pred_id khác
            if gt_id in prev_gt_to_pred and prev_gt_to_pred[gt_id] != pred_id:
                total_id_switch += 1

        prev_gt_to_pred = curr_gt_to_pred

    mota = 1.0 - (total_fp + total_fn + total_id_switch) / max(total_gt, 1)
    motp = total_iou / max(total_matched, 1)

    return {
        'MOTA'      : round(mota * 100, 2),
        'MOTP'      : round(motp * 100, 2),
        'IDs'       : total_id_switch,
        'FP'        : total_fp,
        'FN'        : total_fn,
        'GT'        : total_gt,
        'Matched'   : total_matched,
    }


# ─────────────────────────────────────────────────────────────
# IDF1
# ─────────────────────────────────────────────────────────────

def _compute_idf1(
    gt_per_frame:   Dict[int, List],
    pred_per_frame: Dict[int, List],
    iou_thresh:     float
) -> float:
    """
    Tính IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    IDTP: số frame mà đúng identity được assigned
    IDFP: số pred frame với identity sai
    IDFN: số gt frame không được covered
    """
    # Xây dựng bảng gt_id → list frame, pred_id → list frame
    gt_id_frames:   Dict[int, set] = defaultdict(set)
    pred_id_frames: Dict[int, set] = defaultdict(set)

    all_frames = sorted(set(gt_per_frame) | set(pred_per_frame))

    # Map (frame, gt_id) → pred_id (matched)
    frame_gt_to_pred: Dict[Tuple, int] = {}

    for frame_id in all_frames:
        gt_boxes   = gt_per_frame.get(frame_id, [])
        pred_boxes = pred_per_frame.get(frame_id, [])

        for g in gt_boxes:
            gt_id_frames[int(g[4])].add(frame_id)
        for p in pred_boxes:
            pred_id_frames[int(p[4])].add(frame_id)

        if not gt_boxes or not pred_boxes:
            continue

        matches, _, _ = _match_detections(
            [g[:4] for g in gt_boxes],
            [p[:4] for p in pred_boxes],
            iou_thresh
        )
        for gi, pi in matches:
            gt_id   = int(gt_boxes[gi][4])
            pred_id = int(pred_boxes[pi][4])
            frame_gt_to_pred[(frame_id, gt_id)] = pred_id

    # Với mỗi gt_id, tìm pred_id tốt nhất (xuất hiện nhiều nhất)
    idtp = 0
    idfp_total = 0
    idfn_total = 0

    gt_ids = list(gt_id_frames.keys())
    for gt_id in gt_ids:
        gt_frames = gt_id_frames[gt_id]

        # Đếm số lần gt_id được match với từng pred_id
        pred_count: Dict[int, int] = defaultdict(int)
        for f in gt_frames:
            if (f, gt_id) in frame_gt_to_pred:
                pred_count[frame_gt_to_pred[(f, gt_id)]] += 1

        if not pred_count:
            idfn_total += len(gt_frames)
            continue

        # Chọn pred_id match nhiều nhất
        best_pred_id = max(pred_count, key=pred_count.get)
        tp = pred_count[best_pred_id]
        idtp += tp

        # IDFN: số frame gt_id không được match với best_pred_id
        idfn_total += len(gt_frames) - tp

        # IDFP: số frame best_pred_id xuất hiện nhưng không match gt_id
        pred_frames = pred_id_frames[best_pred_id]
        idfp_total += len(pred_frames) - tp

    idf1 = 2 * idtp / max(2 * idtp + idfp_total + idfn_total, 1)
    return round(idf1 * 100, 2)


# ─────────────────────────────────────────────────────────────
# HOTA
# ─────────────────────────────────────────────────────────────

def _compute_hota(
    gt_per_frame:   Dict[int, List],
    pred_per_frame: Dict[int, List],
    iou_thresholds: List[float] = None
) -> Dict:
    """
    Tính HOTA theo paper: https://arxiv.org/abs/2009.14711

    HOTA = sqrt(DetA * AssA)
    DetA = DetTP / (DetTP + DetFP + DetFN)
    AssA = mean(AssTP / (AssTP + AssFP + AssFN)) qua từng matched pair
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.05, 0.95 + 1e-6, 0.05).tolist()

    hota_scores = []

    for alpha in iou_thresholds:
        det_tp = 0
        det_fp = 0
        det_fn = 0

        # (gt_id, pred_id) → số lần match
        ass_count: Dict[Tuple, int] = defaultdict(int)
        gt_id_total:   Dict[int, int] = defaultdict(int)
        pred_id_total: Dict[int, int] = defaultdict(int)

        all_frames = sorted(set(gt_per_frame) | set(pred_per_frame))

        for frame_id in all_frames:
            gt_boxes   = gt_per_frame.get(frame_id, [])
            pred_boxes = pred_per_frame.get(frame_id, [])

            for g in gt_boxes:
                gt_id_total[int(g[4])] += 1
            for p in pred_boxes:
                pred_id_total[int(p[4])] += 1

            if not gt_boxes and not pred_boxes:
                continue

            matches, unmatched_gt, unmatched_pred = _match_detections(
                [g[:4] for g in gt_boxes],
                [p[:4] for p in pred_boxes],
                alpha
            )

            det_tp += len(matches)
            det_fn += len(unmatched_gt)
            det_fp += len(unmatched_pred)

            for gi, pi in matches:
                gt_id   = int(gt_boxes[gi][4])
                pred_id = int(pred_boxes[pi][4])
                ass_count[(gt_id, pred_id)] += 1

        # DetA
        det_a = det_tp / max(det_tp + det_fp + det_fn, 1)

        # AssA: với mỗi matched pair (gt_id, pred_id)
        ass_scores = []
        for (gt_id, pred_id), tp_ass in ass_count.items():
            fn_ass = gt_id_total[gt_id]   - tp_ass
            fp_ass = pred_id_total[pred_id] - tp_ass
            ass_scores.append(tp_ass / max(tp_ass + fn_ass + fp_ass, 1))

        ass_a = np.mean(ass_scores) if ass_scores else 0.0
        hota_scores.append(np.sqrt(det_a * ass_a))

    hota = float(np.mean(hota_scores))
    return round(hota * 100, 2)


# ─────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    all_tracks:  Dict,
    csv_path:    str,
    iou_thresh:  float = 0.5,
    verbose:     bool  = True,
) -> Dict:
    """
    Tính đầy đủ các độ đo MOT.

    Parameters
    ----------
    all_tracks : dict — output của run_mot() hoặc run_and_visualize()
    csv_path   : str  — đường dẫn file annotation CSV
    iou_thresh : float — ngưỡng IoU để match GT-Pred (mặc định 0.5)
    verbose    : bool  — in kết quả ra màn hình

    Returns
    -------
    dict với các key: MOTA, MOTP, IDF1, HOTA, IDs, FP, FN, GT
    """
    print("Đang đọc Ground Truth...")
    gt_per_frame   = load_gt_from_csv(csv_path)
    pred_per_frame = all_tracks_to_per_frame(all_tracks)

    print(f"GT : {len(gt_per_frame)} frames, "
          f"{sum(len(v) for v in gt_per_frame.values())} objects")
    print(f"Pred: {len(pred_per_frame)} frames, "
          f"{sum(len(v) for v in pred_per_frame.values())} objects")

    print("Tính MOTA / MOTP / IDs...")
    mota_dict = _compute_mota_motp(gt_per_frame, pred_per_frame, iou_thresh)

    print("Tính IDF1...")
    idf1 = _compute_idf1(gt_per_frame, pred_per_frame, iou_thresh)

    print("Tính HOTA...")
    hota = _compute_hota(gt_per_frame, pred_per_frame)

    results = {
        'MOTA' : mota_dict['MOTA'],
        'MOTP' : mota_dict['MOTP'],
        'IDF1' : idf1,
        'HOTA' : hota,
        'IDs'  : mota_dict['IDs'],
        'FP'   : mota_dict['FP'],
        'FN'   : mota_dict['FN'],
        'GT'   : mota_dict['GT'],
    }

    if verbose:
        print("\n" + "=" * 45)
        print("       KẾT QUẢ ĐÁNH GIÁ MOT PIPELINE")
        print("=" * 45)
        print(f"  MOTA  : {results['MOTA']:>8.2f} %")
        print(f"  MOTP  : {results['MOTP']:>8.2f} %")
        print(f"  IDF1  : {results['IDF1']:>8.2f} %")
        print(f"  HOTA  : {results['HOTA']:>8.2f} %")
        print(f"  IDs   : {results['IDs']:>8d}")
        print(f"  FP    : {results['FP']:>8d}")
        print(f"  FN    : {results['FN']:>8d}")
        print(f"  GT    : {results['GT']:>8d}")
        print("=" * 45)

    return results