"""
ByteTrack — Multi-Object Tracking by Associating Every Detection Box
Implement theo paper: https://arxiv.org/abs/2110.06864
Dựa trên repo gốc: https://github.com/FoundationVision/ByteTrack

Input : list of dict [{'tlbr': [x1,y1,x2,y2], 'score': float, 'feat': np.array}, ...]
Output: List[STrack]  — các track đang active, mỗi track có .track_id và .tlbr
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import lap
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter


# ─────────────────────────────────────────────────────────────
# STrack — đại diện cho một tracklet trong ByteTrack
# ─────────────────────────────────────────────────────────────

class STrack(BaseTrack):
    """
    Single object track state — dùng Kalman Filter để dự đoán vị trí.
    State vector: [cx, cy, aspect_ratio, h, vx, vy, va, vh]  (xyah)
    """

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float):
        self._tlwh        = np.asarray(tlwh, dtype=np.float64)
        self.score        = score
        self.is_activated = False
        self.tracklet_len = 0

        # Kalman state
        self.mean         = None
        self.covariance   = None

        # Kalman filter instance — gán khi activate
        self.kalman_filter: Optional[KalmanFilter] = None

        # Appearance feature
        self.feat         = None
        self.smooth_feat  = None

    # ── Kalman predict ──────────────────────────────────────

    def predict(self):
        """Single-track predict — chỉ gọi khi mean đã được khởi tạo."""
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0   # vh = 0 khi không phải Tracked
        self.mean, self.covariance = self.shared_kalman.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks: List['STrack']):
        """Vectorized Kalman predict — chỉ predict track đã có mean."""
        if not stracks:
            return
        # ── Robustness: chỉ predict track đã initiate ──
        valid = [st for st in stracks if st.mean is not None]
        if not valid:
            return

        multi_mean       = np.asarray([st.mean.copy() for st in valid])
        multi_covariance = np.asarray([st.covariance     for st in valid])

        for i, st in enumerate(valid):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0   # vh

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            valid[i].mean       = mean
            valid[i].covariance = cov

    # ── Lifecycle ───────────────────────────────────────────

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Khởi tạo track mới lần đầu."""
        self.kalman_filter = kalman_filter
        self.track_id      = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state        = TrackState.Tracked
        self.is_activated = (frame_id == 1)   # frame 1 → confirmed ngay
        self.frame_id     = frame_id
        self.start_frame  = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """Tái kích hoạt track từ Lost."""
        # ── Robustness: đảm bảo kalman_filter đã được gán ──
        if self.kalman_filter is None:
            self.kalman_filter = STrack.shared_kalman
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.score        = new_track.score
        if new_id:
            self.track_id = self.next_id()
        if new_track.feat is not None:
            self.update_features(new_track.feat)

    def update(self, new_track: 'STrack', frame_id: int):
        """Update track với detection mới."""
        # ── Robustness: đảm bảo kalman_filter đã được gán ──
        if self.kalman_filter is None:
            self.kalman_filter = STrack.shared_kalman
        self.frame_id      = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.score        = new_track.score
        if new_track.feat is not None:
            self.update_features(new_track.feat)

    # ── Feature EMA ─────────────────────────────────────────

    def update_features(self, feat: np.ndarray, alpha: float = 0.9):
        """EMA update cho appearance feature (normalize sau mỗi lần update)."""
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return
        feat = feat / norm
        self.feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
            s_norm = np.linalg.norm(self.smooth_feat)
            if s_norm > 1e-8:
                self.smooth_feat /= s_norm

    # ── BBox converters ─────────────────────────────────────

    @property
    def tlwh(self) -> np.ndarray:
        """[x1, y1, w, h] từ Kalman mean hoặc init value."""
        if self.mean is None:
            return self._tlwh.copy()
        ret    = self.mean[:4].copy()
        ret[2] = ret[2] * ret[3]    # aspect * h → w
        ret[:2] -= ret[2:] / 2      # cx,cy → x1,y1
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """[x1, y1, x2, y2]"""
        ret      = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """[x1,y1,w,h] → [cx, cy, aspect_ratio, h]"""
        ret      = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2      # → cx, cy
        ret[2]  /= ret[3]           # w/h → aspect ratio
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """[x1,y1,w,h] → [cx,cy,w,h]"""
        ret      = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret      = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
        ret      = np.asarray(tlwh, dtype=np.float64).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def _iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU matrix giữa hai tập bbox [x1,y1,x2,y2]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    ix1   = np.maximum(boxes1[:, 0, None], boxes2[None, :, 0])
    iy1   = np.maximum(boxes1[:, 1, None], boxes2[None, :, 1])
    ix2   = np.minimum(boxes1[:, 2, None], boxes2[None, :, 2])
    iy2   = np.minimum(boxes1[:, 3, None], boxes2[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-8
    return inter / union


def iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """Cost matrix dựa trên IoU (1 - IoU)."""
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    aboxes = np.array([t.tlbr for t in atracks], dtype=np.float32)
    bboxes = np.array([t.tlbr for t in btracks], dtype=np.float32)
    return 1.0 - _iou_batch(aboxes, bboxes)


def embedding_distance(
    atracks: List[STrack],
    btracks: List[STrack],
    metric: str = 'cosine'
) -> np.ndarray:
    """Cost matrix dựa trên appearance feature."""
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    # ── Robustness: fallback về IoU nếu feat chưa có ──
    a_feats = [t.smooth_feat for t in atracks]
    b_feats = [t.smooth_feat for t in btracks]
    if any(f is None for f in a_feats) or any(f is None for f in b_feats):
        return iou_distance(atracks, btracks)

    A = np.array(a_feats, dtype=np.float32)
    B = np.array(b_feats, dtype=np.float32)
    if metric == 'cosine':
        cost_matrix = 1.0 - np.dot(A, B.T)
    elif metric == 'euclidean':
        cost_matrix = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: List[STrack]) -> np.ndarray:
    """
    Fuse IoU cost với detection score:
        fused_cost = 1 - (1 - iou_cost) * score
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim    = 1.0 - cost_matrix
    det_scores = np.array([d.score for d in detections], dtype=np.float32)
    fused_sim  = iou_sim * det_scores[None, :]
    return 1.0 - fused_sim


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching dùng lapjv."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1])
        )
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches     = np.array([[i, x[i]] for i in range(len(x)) if x[i] >= 0], dtype=int)
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    if matches.size == 0:
        matches = np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def joint_stracks(lista: List[STrack], listb: List[STrack]) -> List[STrack]:
    exists = {t.track_id for t in lista}
    return lista + [t for t in listb if t.track_id not in exists]


def sub_stracks(lista: List[STrack], listb: List[STrack]) -> List[STrack]:
    remove = {t.track_id for t in listb}
    return [t for t in lista if t.track_id not in remove]


def remove_duplicate_stracks(
    stracksa: List[STrack],
    stracksb: List[STrack],
    iou_thresh: float = 0.15
) -> Tuple[List[STrack], List[STrack]]:
    if not stracksa or not stracksb:
        return stracksa, stracksb
    aboxes = np.array([t.tlbr for t in stracksa], dtype=np.float32)
    bboxes = np.array([t.tlbr for t in stracksb], dtype=np.float32)
    iou    = _iou_batch(aboxes, bboxes)
    pairs  = np.where(iou > iou_thresh)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        len_a = stracksa[p].frame_id - stracksa[p].start_frame
        len_b = stracksb[q].frame_id - stracksb[q].start_frame
        if len_a > len_b:
            dupb.add(q)
        else:
            dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


# ─────────────────────────────────────────────────────────────
# ByteTrack — main class
# ─────────────────────────────────────────────────────────────

class ByteTrack:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    Association flow mỗi frame:
      1. Tách tracked → confirmed / unconfirmed
      2. First assoc  : (confirmed + lost) ↔ high dets  [fuse score]
      3. Second assoc : (unmatched confirmed, Tracked) ↔ low dets
      4. Third assoc  : unconfirmed ↔ remaining high dets
      5. Tạo track mới từ detection còn dư

    Input: list[{'tlbr': [x1,y1,x2,y2], 'score': float, 'feat': np.ndarray}]
    Output: List[STrack]
    """

    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh:  float = 0.1,
        new_track_thresh:  float = 0.65,
        track_buffer:      int   = 60,
        match_thresh:      float = 0.8,
        frame_rate:        int   = 30,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.new_track_thresh  = new_track_thresh
        self.match_thresh      = match_thresh
        self.max_time_lost     = int(frame_rate / 30.0 * track_buffer)

        self.kalman_filter     = KalmanFilter()

        self.tracked_stracks:  List[STrack] = []
        self.lost_stracks:     List[STrack] = []
        self.removed_stracks:  List[STrack] = []

        self.frame_id = 0
        BaseTrack.clear_count()

    # ── Detection helper ─────────────────────────────────────

    @staticmethod
    def _build_stracks(det_list: List[Dict]) -> List[STrack]:
        """Chuyển list dict detection → list STrack (chưa activate)."""
        out = []
        for d in det_list:
            x1, y1, x2, y2 = d['tlbr']
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)
            t    = STrack(tlwh, float(d['score']))
            if d.get('feat') is not None:
                t.update_features(d['feat'])
            out.append(t)
        return out

    # ── Main update ──────────────────────────────────────────

    def update(self, detections: List[Dict], frame_id: int) -> List[STrack]:
        """
        Chạy một bước ByteTrack cho frame hiện tại.

        Parameters
        ----------
        detections : list of dict  {'tlbr', 'score', 'feat'(optional)}
        frame_id   : int (1-based)

        Returns
        -------
        List[STrack] — các track đang active (is_activated=True)
        """
        self.frame_id = frame_id

        activated_stracks: List[STrack] = []
        refind_stracks:    List[STrack] = []
        lost_stracks:      List[STrack] = []
        removed_stracks:   List[STrack] = []

        # ── 0. Phân loại detection ───────────────────────────
        det_high = [d for d in detections if d['score'] >= self.track_high_thresh]
        det_low  = [d for d in detections
                    if self.track_low_thresh <= d['score'] < self.track_high_thresh]

        dets_high = self._build_stracks(det_high)
        dets_low  = self._build_stracks(det_low)

        # ── 1. Tách confirmed / unconfirmed ─────────────────
        #    *** ĐÂY LÀ BƯỚC BỊ THIẾU TRƯỚC ĐÓ ***
        #    confirmed   = đã từng được activate (is_activated=True)
        #    unconfirmed = mới tạo frame trước, chưa confirmed
        confirmed   = [t for t in self.tracked_stracks if     t.is_activated]
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]

        # ── 2. Kalman predict toàn bộ pool ──────────────────
        #    pool = confirmed + lost  (unconfirmed cũng được predict)
        strack_pool = joint_stracks(confirmed, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        STrack.multi_predict(unconfirmed)   # predict riêng unconfirmed

        # ── 3. First association: (confirmed+lost) ↔ high dets
        dists_1 = iou_distance(strack_pool, dets_high)
        dists_1 = fuse_score(dists_1, dets_high)          # fuse detection score
        matches_1, u_track_1, u_det_1 = linear_assignment(dists_1, self.match_thresh)

        for itracked, idet in matches_1:
            track = strack_pool[itracked]
            det   = dets_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_stracks.append(track)
            else:                                          # Lost → re-find
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        # ── 4. Second association: unmatched confirmed ↔ low dets
        #    Chỉ lấy track Tracked state (không lấy Lost đã re-find ở trên)
        r_tracked = [strack_pool[i] for i in u_track_1
                     if strack_pool[i].state == TrackState.Tracked]

        dists_2 = iou_distance(r_tracked, dets_low)
        # NOTE: second assoc KHÔNG fuse score (ByteTrack gốc)
        matches_2, u_track_2, _ = linear_assignment(dists_2, thresh=0.5)

        for itracked, idet in matches_2:
            track = r_tracked[itracked]
            det   = dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        # Track không match lần nào → Lost
        for i in u_track_2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── 5. Third association: unconfirmed ↔ remaining high dets
        #    *** ĐÚNG: unconfirmed match với high dets còn lại sau first assoc ***
        rem_dets = [dets_high[i] for i in u_det_1]

        dists_unc                       = iou_distance(unconfirmed, rem_dets)
        dists_unc                       = fuse_score(dists_unc, rem_dets)
        matches_unc, u_unconf, u_det_2  = linear_assignment(dists_unc, thresh=0.7)

        for itracked, idet in matches_unc:
            unconfirmed[itracked].update(rem_dets[idet], frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # Unconfirmed không match → remove ngay (chưa đủ tuổi)
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            removed_stracks.append(unconfirmed[i])

        # ── 6. Tạo track mới ─────────────────────────────────
        for i in u_det_2:
            det = rem_dets[i]
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, frame_id)
                activated_stracks.append(det)

        # ── 7. Lost → Removed nếu quá hạn ───────────────────
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ── 8. Cập nhật pool ──────────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                 if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)
        self.removed_stracks = self.removed_stracks[-1000:]   # tránh memory leak

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return [t for t in self.tracked_stracks if t.is_activated]