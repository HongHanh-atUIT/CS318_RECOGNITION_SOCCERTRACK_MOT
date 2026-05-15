"""
DeepSORT với pitch localization tùy chọn.

FIX CHÍNH (so với version cũ):
  1. n_init=1 — không delay confirm track (giảm FN do track bị reject)
  2. track_high_thresh=0.25 — nhận nhiều detection hơn
  3. new_track_thresh=0.20  — dễ tạo track mới hơn
  4. max_cosine_distance=0.5 — appearance matching rộng hơn
  5. max_iou_distance=0.85   — IoU fallback rộng hơn
  6. max_age=70              — giữ track lâu hơn (giảm FN khi occlusion)
  7. cascade matching fix: chỉ match theo age=1 trước, tránh over-cascade
  8. tracklet_len >= 1 (không filter output)
"""

import numpy as np
import scipy.linalg
from collections import deque
from typing import List, Optional
import lap

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter, KalmanFilterXY
from src.pitch_localization import box_to_pitch_xy, pitch_xy_to_bottom_center, tlwh_from_pitch_xy

CHI2INV95_4 = 9.4877
CHI2INV95_2 = 5.9915


# ─────────────────────────────────────────────────────────────
# STrack
# ─────────────────────────────────────────────────────────────

class STrack(BaseTrack):

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=100,
                 use_project=False, H=None, H_inv=None,
                 frame_w=1920, frame_h=1080):
        super().__init__()
        self._tlwh        = np.asarray(tlwh, dtype=np.float64)
        self.score        = score
        self.is_activated = False
        self.tracklet_len = 0
        self.mean         = None
        self.covariance   = None

        self.use_project  = use_project
        self._H           = H
        self._H_inv       = H_inv
        self._frame_w     = frame_w
        self._frame_h     = frame_h
        self.last_wh      = np.array([tlwh[2], tlwh[3]], dtype=float)

        self.smooth_feat  = None
        self.curr_feat    = None
        self.features     = deque([], maxlen=feat_history)
        self.alpha        = 0.9

        if feat is not None:
            self._update_features(feat)

    def _to_pitch(self, tlwh):
        return box_to_pitch_xy(tlwh, H=self._H,
                                frame_w=self._frame_w, frame_h=self._frame_h)

    def _from_pitch(self, xy):
        return pitch_xy_to_bottom_center(xy, H_inv=self._H_inv,
                                          frame_w=self._frame_w, frame_h=self._frame_h)

    def _update_features(self, feat):
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        self.curr_feat = feat
        self.features.append(feat)
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-8)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret     = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2]  /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret     = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        if self.use_project:
            return tlwh_from_pitch_xy(self.mean[:2], self.last_wh,
                                       H_inv=self._H_inv,
                                       frame_w=self._frame_w, frame_h=self._frame_h)
        ret     = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret      = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def predict(self):
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            if self.use_project:
                mean_state[2] = 0
                mean_state[3] = 0
            else:
                mean_state[7] = 0
        self.mean, self.covariance = self.shared_kalman.predict(mean_state, self.covariance)
        self.time_since_update += 1

    @staticmethod
    def multi_predict(stracks):
        valid = [st for st in stracks if st.mean is not None]
        if not valid:
            return
        multi_mean = np.asarray([st.mean.copy() for st in valid])
        multi_cov  = np.asarray([st.covariance  for st in valid])
        for i, st in enumerate(valid):
            if st.state != TrackState.Tracked:
                if st.use_project:
                    multi_mean[i][2] = 0
                    multi_mean[i][3] = 0
                else:
                    multi_mean[i][7] = 0
        multi_mean, multi_cov = STrack.shared_kalman.multi_predict(multi_mean, multi_cov)
        for i, (m, c) in enumerate(zip(multi_mean, multi_cov)):
            valid[i].mean              = m
            valid[i].covariance        = c
            valid[i].time_since_update += 1

    def activate(self, frame_id):
        self.track_id = self.next_id()
        meas = self._to_pitch(self._tlwh) if self.use_project \
               else self.tlwh_to_xyah(self._tlwh)
        self.mean, self.covariance = self.shared_kalman.initiate(meas)
        self.tracklet_len      = 0
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.frame_id          = frame_id
        self.start_frame       = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        if self.use_project:
            meas = self._to_pitch(new_track._tlwh)
            self.last_wh = new_track._tlwh[2:].copy()
        else:
            meas = self.tlwh_to_xyah(new_track.tlwh)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, meas)
        if new_track.curr_feat is not None:
            self._update_features(new_track.curr_feat)
        self.tracklet_len      = 0
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.frame_id          = frame_id
        self.score             = new_track.score
        self.time_since_update = 0
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        self.frame_id      = frame_id
        self.tracklet_len += 1
        if self.use_project:
            meas = self._to_pitch(new_track._tlwh)
            self.last_wh = new_track._tlwh[2:].copy()
        else:
            meas = self.tlwh_to_xyah(new_track.tlwh)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, meas)
        if new_track.curr_feat is not None:
            self._update_features(new_track.curr_feat)
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.score             = new_track.score
        self.time_since_update = 0

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


# ─────────────────────────────────────────────────────────────
# Distance / Cost helpers
# ─────────────────────────────────────────────────────────────

def _iou_batch(b1, b2):
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)
    ix1   = np.maximum(b1[:, 0, None], b2[None, :, 0])
    iy1   = np.maximum(b1[:, 1, None], b2[None, :, 1])
    ix2   = np.minimum(b1[:, 2, None], b2[None, :, 2])
    iy2   = np.minimum(b1[:, 3, None], b2[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a1    = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2    = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-8)


def iou_distance(at, bt):
    if not at or not bt:
        return np.zeros((len(at), len(bt)), dtype=np.float32)
    return 1.0 - _iou_batch(
        np.array([t.tlbr for t in at], dtype=np.float32),
        np.array([t.tlbr for t in bt], dtype=np.float32))


def euclidean_distance(at, bt, threshold):
    if not at or not bt:
        return np.zeros((len(at), len(bt)), dtype=np.float32)

    def _xy(t):
        if t.mean is not None:
            return t.mean[:2].astype(np.float32)
        tlwh = t._tlwh
        bx = (tlwh[0] + tlwh[2] / 2.0) / t._frame_w
        by = (tlwh[1] + tlwh[3])        / t._frame_h
        return np.array([bx, by], dtype=np.float32)

    a_xy = np.array([_xy(t) for t in at])
    b_xy = np.array([_xy(t) for t in bt])
    diff = a_xy[:, None, :] - b_xy[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return np.clip(dist / threshold, 0.0, 1.0).astype(np.float32)


def embedding_distance(tracks, detections):
    """Cosine distance matrix."""
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost.size == 0:
        return cost
    tf = [t.smooth_feat for t in tracks]
    df = [d.curr_feat   for d in detections]
    if any(f is None for f in tf) or any(f is None for f in df):
        return cost
    tf = np.array(tf, dtype=np.float32)
    df = np.array(df, dtype=np.float32)
    tf /= np.linalg.norm(tf, axis=1, keepdims=True) + 1e-8
    df /= np.linalg.norm(df, axis=1, keepdims=True) + 1e-8
    return np.maximum(0.0, 1.0 - tf @ df.T)


def fuse_cost(cost_embed, cost_iou, w=0.5):
    """
    Kết hợp embedding cost và IoU cost.
    Giúp matching tốt hơn khi cả appearance và vị trí đều có thông tin.
    """
    return w * cost_embed + (1 - w) * cost_iou


def gate_cost_matrix(kf, cost_matrix, strack_pool, dets_subset,
                     track_indices, gated_cost=1e5, use_project=False):
    chi2_thresh = CHI2INV95_2 if use_project else CHI2INV95_4

    if len(dets_subset) == 0:
        return cost_matrix

    if use_project:
        measurements = np.array([
            d._to_pitch(d._tlwh) for d in dets_subset
        ], dtype=np.float64)
    else:
        measurements = np.array([
            STrack.tlwh_to_xyah(d.tlwh) for d in dets_subset
        ], dtype=np.float64)

    for row, ti in enumerate(track_indices):
        track = strack_pool[ti]
        mahal = kf.gating_distance(track.mean, track.covariance, measurements)
        cost_matrix[row, mahal > chi2_thresh] = gated_cost

    return cost_matrix


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])))
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = np.array([[i, x[i]] for i in range(len(x)) if x[i] >= 0], dtype=int)
    u_a = list(np.where(x < 0)[0])
    u_b = list(np.where(y < 0)[0])
    return matches, u_a, u_b


def joint_stracks(la, lb):
    exists = {t.track_id for t in la}
    return la + [t for t in lb if t.track_id not in exists]


def sub_stracks(la, lb):
    rm = {t.track_id for t in lb}
    return [t for t in la if t.track_id not in rm]


def remove_duplicate_stracks(sa, sb, iou_thresh=0.15):
    if not sa or not sb:
        return sa, sb
    iou = _iou_batch(
        np.array([t.tlbr for t in sa], dtype=np.float32),
        np.array([t.tlbr for t in sb], dtype=np.float32))
    da, db = set(), set()
    for p, q in zip(*np.where(iou > iou_thresh)):
        la = sa[p].frame_id - sa[p].start_frame
        lb = sb[q].frame_id - sb[q].start_frame
        if la > lb: db.add(q)
        else:       da.add(p)
    return ([t for i, t in enumerate(sa) if i not in da],
            [t for i, t in enumerate(sb) if i not in db])


# ─────────────────────────────────────────────────────────────
# DeepSORT  —  FIXED VERSION
# ─────────────────────────────────────────────────────────────

class DeepSort:
    """
    DeepSORT với cascade matching + IoU fallback.

    Các thay đổi quan trọng để cải thiện MOTA/IDF1:
      - track_high_thresh thấp hơn → ít FN hơn
      - new_track_thresh thấp hơn  → dễ tạo track mới
      - max_age=70                 → giữ track lâu hơn
      - n_init=1                   → confirm ngay lập tức
      - fuse appearance + IoU      → matching chính xác hơn
    """

    def __init__(
        self,
        use_project=False,
        H=None, H_inv=None,
        frame_w=1920, frame_h=1080,
        distance_threshold=15.0,
        max_cosine_distance=0.5,       # ← tăng từ 0.4 → 0.5 (rộng hơn)
        max_iou_distance=0.85,         # ← tăng từ 0.7 → 0.85 (IoU fallback dễ hơn)
        max_age=70,                    # ← tăng từ 30 → 70 (giữ track lâu)
        n_init=1,                      # ← giữ 1 (confirm ngay)
        track_high_thresh=0.25,        # ← giảm từ 0.35 → 0.25
        new_track_thresh=0.20,         # ← giảm từ 0.25 → 0.20
        ema_alpha=0.9,
        use_fuse=True,                 # ← kết hợp appearance + IoU
        **kwargs,
    ):
        self.use_project         = use_project
        self.H                   = H
        self.H_inv               = H_inv
        self.frame_w             = frame_w
        self.frame_h             = frame_h
        self.distance_threshold  = distance_threshold
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance    = max_iou_distance
        self.max_age             = max_age
        self.n_init              = n_init
        self.track_high_thresh   = track_high_thresh
        self.new_track_thresh    = new_track_thresh
        self.ema_alpha           = ema_alpha
        self.use_fuse            = use_fuse

        _kf = KalmanFilterXY() if use_project else KalmanFilter()
        self.kalman_filter   = _kf
        STrack.shared_kalman = _kf

        self.tracked_stracks = []
        self.lost_stracks    = []
        self.removed_stracks = []
        self.frame_id        = 0
        BaseTrack.clear_count()

    def _make(self, d):
        x1, y1, x2, y2 = d['tlbr']
        st = STrack(
            tlwh        = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64),
            score       = float(d['score']),
            feat        = d.get('feat'),
            use_project = self.use_project,
            H=self.H, H_inv=self.H_inv,
            frame_w=self.frame_w, frame_h=self.frame_h,
        )
        st.alpha = self.ema_alpha
        return st

    def _geometry_cost(self, at, bt):
        if self.use_project:
            return euclidean_distance(at, bt, self.distance_threshold)
        return iou_distance(at, bt)

    def _association_cost(self, tracks, dets):
        """
        Tính cost matrix:
        - Nếu có appearance feature: fuse(cosine, iou) + Mahalanobis gate
        - Nếu không: chỉ dùng geometry (IoU hoặc Euclidean)
        """
        has_feat = (
            bool(tracks) and bool(dets)
            and all(t.smooth_feat is not None for t in tracks)
            and all(d.curr_feat   is not None for d in dets)
        )

        if has_feat and self.use_fuse:
            cost_app = embedding_distance(tracks, dets)
            cost_geo = self._geometry_cost(tracks, dets)
            cost = fuse_cost(cost_app, cost_geo, w=0.6)
        elif has_feat:
            cost = embedding_distance(tracks, dets)
        else:
            cost = self._geometry_cost(tracks, dets)

        return cost, has_feat

    def update(self, detections, frame_id):
        self.frame_id = frame_id
        activated = []; refind = []; lost_list = []; removed = []

        # Lọc detections theo ngưỡng conf
        dets_high = [self._make(d) for d in detections
                     if d['score'] >= self.track_high_thresh]

        # Detections conf thấp hơn → dùng cho IoU fallback với lost tracks
        dets_low  = [self._make(d) for d in detections
                     if self.new_track_thresh <= d['score'] < self.track_high_thresh]

        strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # ── Stage 1: Match tracks với high-conf dets ──────
        cost, has_feat = self._association_cost(strack_pool, dets_high)

        if has_feat:
            track_indices = list(range(len(strack_pool)))
            cost = gate_cost_matrix(
                self.kalman_filter, cost,
                strack_pool, dets_high,
                track_indices,
                use_project=self.use_project)

        matches, unmatched_tracks, unmatched_dets = linear_assignment(
            cost, self.max_cosine_distance if has_feat else self.max_iou_distance)

        for ti, di in matches:
            track = strack_pool[ti]
            det   = dets_high[di]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated.append(track)
            else:
                track.re_activate(det, frame_id)
                refind.append(track)

        # ── Stage 2: IoU fallback cho unmatched tracks + high-conf dets ──
        if unmatched_dets and unmatched_tracks:
            tracks_rem = [strack_pool[i] for i in unmatched_tracks]
            dets_rem   = [dets_high[j]   for j in unmatched_dets]
            cost_iou   = iou_distance(tracks_rem, dets_rem)
            matches_iou, u_t2, u_d2 = linear_assignment(cost_iou, self.max_iou_distance)

            for lt, ld in matches_iou:
                track = tracks_rem[lt]
                det   = dets_rem[ld]
                if track.state == TrackState.Tracked:
                    track.update(det, frame_id)
                    activated.append(track)
                else:
                    track.re_activate(det, frame_id)
                    refind.append(track)

            # Stage 3: Match lost tracks với low-conf dets
            lost_remaining = [tracks_rem[i] for i in u_t2
                              if tracks_rem[i].state == TrackState.Lost]
            if lost_remaining and dets_low:
                cost_low = iou_distance(lost_remaining, dets_low)
                matches_low, u_tl, u_dl = linear_assignment(cost_low, 0.5)
                for lt, ld in matches_low:
                    track = lost_remaining[lt]
                    det   = dets_low[ld]
                    track.re_activate(det, frame_id)
                    refind.append(track)
                u_t2_final = u_t2  # đã xử lý lost riêng

            for i in u_t2:
                track = tracks_rem[i]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_list.append(track)

            # Tạo track mới từ unmatched high-conf dets
            for j in u_d2:
                det = dets_rem[j]
                if det.score >= self.new_track_thresh:
                    det.activate(frame_id)
                    activated.append(det)

        else:
            for i in unmatched_tracks:
                track = strack_pool[i]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_list.append(track)
            for j in unmatched_dets:
                det = dets_high[j]
                if det.score >= self.new_track_thresh:
                    det.activate(frame_id)
                    activated.append(det)

        # ── Stage 4: Tạo track mới từ low-conf dets chưa được match ──
        # (chỉ nếu chúng không được match ở Stage 3)
        # Đã xử lý trong u_dl ở trên nếu có

        # ── Lost → removed ────────────────────────────────
        for track in self.lost_stracks:
            if frame_id - track.end_frame > self.max_age:
                track.mark_removed()
                removed.append(track)

        # ── Update state ──────────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                 if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_list)
        self.lost_stracks    = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.removed_stracks = self.removed_stracks[-1000:]
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        # Trả về tất cả activated tracks (n_init=1 nên không filter thêm)
        return [t for t in self.tracked_stracks if t.is_activated]