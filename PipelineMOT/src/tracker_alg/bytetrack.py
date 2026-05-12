"""
ByteTrack — Multi-Object Tracking by Associating Every Detection Box
Paper: https://arxiv.org/abs/2110.06864
Repo : https://github.com/FoundationVision/ByteTrack

Mở rộng: thêm Appearance Feature (EMA) vào first association.
  - Nếu feat=None → chạy ByteTrack thuần IoU như gốc
  - Nếu feat có → fuse IoU cost + embedding cosine cost trong first association
Mở rộng 2: thêm lựa chọn sử dụng pitch localization vào mọi bước association/
    - Nếu use_project = True -> thay đổi state từ dạng [x, y, a, h] về [x,y] theo pitch
                                thay cost IOU thành cost theo euclid distance đã biến đổi theo similarity
                                cần điều chỉnh distance_threshold
    - Nếu use_project =  False -> chạy ByteTrack bình thường với IoU

Input : list of dict [{'tlbr': [x1,y1,x2,y2], 'score': float, 'feat': np.array hoặc None}, ...]
Output: List[STrack]

[TODO]: Định nghĩa hàm _tlwh_to_xy_pitch và inverse_homography
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import lap
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilterXY, KalmanFilter

# ─────────────────────────────────────────────────────────────
# STrack
# ─────────────────────────────────────────────────────────────
def inverse_homography(tlwh: np.ndarray):
    # Tự định nghĩa tại đây hoặc import từ module pitch localization
    pass


class STrack(BaseTrack):
    """
    Single object track.
    State vector: [cx, cy, aspect_ratio, h, vx, vy, va, vh]  (xyah)
    """

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, use_project = False):
        super().__init__()  # time_since_update là instance var
        self._tlwh        = np.asarray(tlwh, dtype=np.float64)
        self.score        = score
        self.is_activated = False
        self.tracklet_len = 0
        self.mean         = None
        self.covariance   = None
        self.kalman_filter: Optional[Union[KalmanFilter, KalmanFilterXY]] = None
        
        self.use_project = use_project
        self.last_wh = np.array([tlwh[2], tlwh[3]])  # w, h từ tlwh

        # Appearance feature
        self.feat        = None
        self.smooth_feat = None

    # ── EMA Feature ─────────────────────────────────────────

    def update_features(self, feat: np.ndarray, alpha: float = 0.9):
        """EMA update cho appearance feature."""
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

    # ── Kalman predict ───────────────────────────────────────

    def predict(self):
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            if self.use_project:
                mean_state[2] = 0   # vx = 0
                mean_state[3] = 0   # vy = 0
            else:
                mean_state[7] = 0   # vh = 0
        self.mean, self.covariance = self.shared_kalman.predict(
            mean_state, self.covariance)
        self.time_since_update += 1

    @staticmethod
    def multi_predict(stracks: List['STrack']):
        valid = [st for st in stracks if st.mean is not None]
        if not valid:
            return
        multi_mean       = np.asarray([st.mean.copy() for st in valid])
        multi_covariance = np.asarray([st.covariance  for st in valid])
        
        for i, st in enumerate(valid):
            if st.state != TrackState.Tracked:
                if st.use_project:
                    multi_mean[i][2] = 0  # vx
                    multi_mean[i][3] = 0  # vy
                else:
                    multi_mean[i][7] = 0  # vh
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (m, c) in enumerate(zip(multi_mean, multi_covariance)):
            valid[i].mean              = m
            valid[i].covariance        = c
            valid[i].time_since_update += 1

    # ── Lifecycle ────────────────────────────────────────────

    def activate(self, kalman_filter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id      = self.next_id()
        if self.use_project:
            measurement = self._tlwh_to_xy_pitch(self._tlwh)     # Định nghĩa hàm _tlwh_to_xy_pitch
        else:
            measurement = self.tlwh_to_xyah(self._tlwh)
            
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)
        self.tracklet_len      = 0
        self.state             = TrackState.Tracked
        self.is_activated      = (frame_id == 1)
        self.frame_id          = frame_id
        self.start_frame       = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        if self.kalman_filter is None:
            self.kalman_filter = STrack.shared_kalman
            
        if self.use_project:
            measurement = self._tlwh_to_xy_pitch(new_track._tlwh)  # TODO
            self.last_wh = new_track._tlwh[2:]
        else:
            measurement = self.tlwh_to_xyah(new_track.tlwh)
            
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement)
        
        if new_track.feat is not None:
            self.update_features(new_track.feat)
        self.tracklet_len      = 0
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.frame_id          = frame_id
        self.score             = new_track.score
        self.time_since_update = 0
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: 'STrack', frame_id: int):
        if self.kalman_filter is None:
            self.kalman_filter = STrack.shared_kalman
        self.frame_id          = frame_id
        self.tracklet_len     += 1
        
        if self.use_project:
            measurement = self._tlwh_to_xy_pitch(new_track._tlwh)  # TODO
            self.last_wh = new_track._tlwh[2:]  # cập nhật w,h
        else:
            measurement = self.tlwh_to_xyah(new_track.tlwh)
            
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement)
        
        if new_track.feat is not None:
            self.update_features(new_track.feat)
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.score             = new_track.score
        self.time_since_update = 0

    # ── BBox converters ──────────────────────────────────────

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret    = self.mean[:4].copy()
        ret[2] = ret[2] * ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        if self.use_project:
            cx, cy_bottom = inverse_homography(self.mean[:2])  # TODO
            w, h = self.last_wh
            return np.array([cx - w/2, cy_bottom - h, cx + w/2, cy_bottom])
        ret      = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret      = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2]  /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret      = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def _tlwh_to_xy_pitch(tlwh):
        """TODO: placeholder — gọi homography từ module pitch localization."""
        return np.array([0.0, 0.0])

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


# ─────────────────────────────────────────────────────────────
# Distance / Cost functions
# ─────────────────────────────────────────────────────────────

def _iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    ix1   = np.maximum(boxes1[:, 0, None], boxes2[None, :, 0])
    iy1   = np.maximum(boxes1[:, 1, None], boxes2[None, :, 1])
    ix2   = np.minimum(boxes1[:, 2, None], boxes2[None, :, 2])
    iy2   = np.minimum(boxes1[:, 3, None], boxes2[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-8)


def iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    return 1.0 - _iou_batch(
        np.array([t.tlbr for t in atracks], dtype=np.float32),
        np.array([t.tlbr for t in btracks], dtype=np.float32))

def _euclidean_batch(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Tính ma trận Euclidean distance giữa 2 tập điểm.

    Args:
        points1 (np.ndarray): Shape (M, 2) — [x, y]
        points2 (np.ndarray): Shape (N, 2) — [x, y]

    Returns:
        np.ndarray: Shape (M, N) — distance matrix, [0, +inf)
    """
    if len(points1) == 0 or len(points2) == 0:
        return np.zeros((len(points1), len(points2)), dtype=np.float32)
    diff = points1[:, None, :] - points2[None, :, :]  # (M, N, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))         # (M, N)


def euclidean_distance(atracks: List[STrack], btracks: List[STrack],
                       distance_threshold: float) -> np.ndarray:
    """
    Tính similarity matrix từ Euclidean distance — dùng thay iou_distance khi use_project=True.
    Normalize về [0, 1]: gần → 1, xa → 0, tương đương IoU về chiều optimize.
    """
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    def get_xy(t: STrack) -> np.ndarray:
        if t.mean is not None:
            return t.mean[:2]
        
        # fallback: bottom center từ tlwh [x1, y1, w, h]
        x1, y1, w, h = t._tlwh
        return np.array([x1 + w / 2, y1 + h])  # cx, y_bottom

    a_xy = np.array([get_xy(t) for t in atracks], dtype=np.float32) # (M, 2)
    b_xy = np.array([get_xy(t) for t in btracks], dtype=np.float32) # (N, 2)

    dist = _euclidean_batch(a_xy, b_xy)                                # (M, N)
    similarity = 1 - np.clip(dist / distance_threshold, 0, 1)         # (M, N) [0,1]
    return 1 - similarity                                               

def embedding_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """Cosine distance dựa trên EMA smooth_feat. Trả về None nếu không có feat."""
    if not atracks or not btracks:
        return None
    a_feats = [t.smooth_feat for t in atracks]
    b_feats = [t.smooth_feat for t in btracks]
    if any(f is None for f in a_feats) or any(f is None for f in b_feats):
        return None   # không có feat → caller dùng IoU thuần
    A = np.array(a_feats, dtype=np.float32)
    B = np.array(b_feats, dtype=np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return np.maximum(0.0, 1.0 - A @ B.T)


def fuse_iou_emb(iou_cost: np.ndarray,
                 emb_cost: Optional[np.ndarray],
                 lambda_iou: float = 0.7) -> np.ndarray:
    """
    Fuse IoU/euclid cost và embedding cost.
    lambda_iou=0.98 → ưu tiên IoU, embedding chỉ là tie-breaker.
    Nếu emb_cost=None → trả về iou_cost thuần.
    """
    if emb_cost is None:
        return iou_cost
    return lambda_iou * iou_cost + (1 - lambda_iou) * emb_cost


def fuse_score(cost_matrix: np.ndarray, detections: List[STrack]) -> np.ndarray:
    """Fuse IoU cost với detection score: fused = 1 - (1-cost)*score"""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim    = 1.0 - cost_matrix
    det_scores = np.array([d.score for d in detections], dtype=np.float32)
    return 1.0 - iou_sim * det_scores[None, :]


def linear_assignment(cost_matrix: np.ndarray,
                      thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(cost_matrix.shape[0]),
                np.arange(cost_matrix.shape[1]))
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


def remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack],
                              iou_thresh: float = 0.15) -> Tuple[List[STrack], List[STrack]]:
    if not stracksa or not stracksb:
        return stracksa, stracksb
    iou  = _iou_batch(
        np.array([t.tlbr for t in stracksa], dtype=np.float32),
        np.array([t.tlbr for t in stracksb], dtype=np.float32))
    dupa, dupb = set(), set()
    for p, q in zip(*np.where(iou > iou_thresh)):
        la = stracksa[p].frame_id - stracksa[p].start_frame
        lb = stracksb[q].frame_id - stracksb[q].start_frame
        if la > lb: dupb.add(q)
        else:       dupa.add(p)
    return ([t for i, t in enumerate(stracksa) if i not in dupa],
            [t for i, t in enumerate(stracksb) if i not in dupb])


# ─────────────────────────────────────────────────────────────
# ByteTrack
# ─────────────────────────────────────────────────────────────

class ByteTrack:
    """
    ByteTrack với Appearance Feature tùy chọn.

    - Nếu extractor truyền feat=None → chạy ByteTrack thuần IoU (gốc)
    - Nếu feat có → first association dùng fused cost (IoU + embedding)

    Association flow:
      1. Tách confirmed / unconfirmed
      2. First assoc : (confirmed + lost) ↔ high dets  [fuse score + optional emb]
      3. Second assoc: unmatched confirmed Tracked ↔ low dets  [IoU only]
      4. Third assoc : unconfirmed ↔ remaining high dets  [IoU + fuse score]
      5. Tạo track mới từ high dets còn dư

    Parameters
    ----------
    lambda_iou : float
        Trọng số IoU trong fused cost khi có appearance feature.
        0.98 = ưu tiên IoU, embedding là tie-breaker.
        1.0  = bỏ hoàn toàn embedding (giống ByteTrack gốc).
    """

    def __init__(
        self,
        use_project:       bool = False,
        distance_threshold = 100,   # Tham số điều chỉnh scale của distance
        track_high_thresh: float = 0.6,
        track_low_thresh:  float = 0.1,
        new_track_thresh:  float = 0.65,
        track_buffer:      int   = 60,
        match_thresh:      float = 0.8,
        frame_rate:        int   = 30,
        lambda_iou:        float = 0.98,   # trọng số IoU khi có appearance feat
    ):
        self.use_project = use_project  # Đánh dấu sử dụng project lên tọa độ sân
        if use_project:
            self.kalman_filter = KalmanFilterXY()
            STrack.shared_kalman = self.kalman_filter  # override class variable
        else:
            self.kalman_filter = KalmanFilter()
            STrack.shared_kalman = self.kalman_filter
        
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.new_track_thresh  = new_track_thresh
        self.match_thresh      = match_thresh
        self.max_time_lost     = int(frame_rate / 30.0 * track_buffer)
        self.lambda_iou        = lambda_iou
        self.distance_threshold = distance_threshold

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        BaseTrack.clear_count()

    # ── Detection helper ─────────────────────────────────────

    def _build_stracks(self, det_list: List[Dict]) -> List[STrack]:
        out = []
        for d in det_list:
            x1, y1, x2, y2 = d['tlbr']
            st = STrack(np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64),
                        float(d['score']), use_project=self.use_project)
            if d.get('feat') is not None:
                st.update_features(d['feat'])
            out.append(st)
        return out

    # ── Main update ──────────────────────────────────────────

    def update(self, detections: List[Dict], frame_id: int) -> List[STrack]:
        self.frame_id = frame_id

        activated_stracks: List[STrack] = []
        refind_stracks:    List[STrack] = []
        lost_stracks:      List[STrack] = []
        removed_stracks:   List[STrack] = []

        # ── 0. Phân loại detection ───────────────────────────
        dets_high = self._build_stracks(
            [d for d in detections if d['score'] >= self.track_high_thresh])
        dets_low  = self._build_stracks(
            [d for d in detections
             if self.track_low_thresh <= d['score'] < self.track_high_thresh])

        # ── 1. Tách confirmed / unconfirmed ─────────────────
        confirmed   = [t for t in self.tracked_stracks if     t.is_activated]
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]

        # ── 2. Kalman predict ────────────────────────────────
        strack_pool = joint_stracks(confirmed, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        STrack.multi_predict(unconfirmed)

        # ── 3. First association: (confirmed+lost) ↔ high dets
        # Có appearance feat → fuse IoU + embedding
        # Không có feat      → IoU thuần (ByteTrack gốc)
        emb_cost_1 = embedding_distance(strack_pool, dets_high)  # None nếu không có feat
        if self.use_project:
            cost_1 = euclidean_distance(strack_pool, dets_high, self.distance_threshold)
        else:
            cost_1 = iou_distance(strack_pool, dets_high)

        cost_1     = fuse_iou_emb(cost_1, emb_cost_1, self.lambda_iou)
        cost_1     = fuse_score(cost_1, dets_high)   # fuse detection score

        matches_1, u_track_1, u_det_1 = linear_assignment(cost_1, self.match_thresh)

        for it, id_ in matches_1:
            track = strack_pool[it]
            det   = dets_high[id_]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        # ── 4. Second association: unmatched Tracked ↔ low dets
        # Second association chỉ dùng IoU (ByteTrack gốc, không fuse emb/score)
        r_tracked = [strack_pool[i] for i in u_track_1
                     if strack_pool[i].state == TrackState.Tracked]
        if self.use_project:
            cost_2 = euclidean_distance(r_tracked, dets_low, self.distance_threshold)
        else:
            cost_2 = iou_distance(r_tracked, dets_low)
        matches_2, u_track_2, _ = linear_assignment(cost_2, thresh=0.5)

        for it, id_ in matches_2:
            track = r_tracked[it]
            det   = dets_low[id_]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        for i in u_track_2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── 5. Third association: unconfirmed ↔ remaining high dets
        rem_dets  = [dets_high[i] for i in u_det_1]
        if self.use_project:
            cost_unc = euclidean_distance(unconfirmed, rem_dets, self.distance_threshold)
        else:
            cost_unc = iou_distance(unconfirmed, rem_dets)
        cost_unc  = fuse_score(cost_unc, rem_dets)
        matches_unc, u_unconf, u_det_2 = linear_assignment(cost_unc, thresh=0.7)

        for it, id_ in matches_unc:
            unconfirmed[it].update(rem_dets[id_], frame_id)
            activated_stracks.append(unconfirmed[it])

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

        # ── 8. Cập nhật pool ─────────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                 if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)
        self.removed_stracks = self.removed_stracks[-1000:]

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]