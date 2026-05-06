import numpy as np
import scipy.linalg
from typing import List, Dict, Optional
from collections import deque
import lap

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter

# Chi-square 0.95 quantile dùng cho Mahalanobis gating (4 DOF = xyah)
CHI2INV95_4 = 9.4877


# ─────────────────────────────────────────────────────────────
# Kalman Filter với NSA — theo đúng code gốc dyhBUPT
# ─────────────────────────────────────────────────────────────

class NSAKalmanFilter(KalmanFilter):
    """
    NSA theo code gốc: scale std TRƯỚC khi tạo innovation_cov trong project().
    std = [(1 - confidence) * x for x in std]
    Cách này đảm bảo R = diag(std²) ≥ 0 → projected_cov luôn positive definite.
    """

    def project(self, mean, covariance, confidence=0.0):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        # NSA: scale std theo (1 - confidence) — đúng theo gốc
        std            = [(1 - confidence) * x for x in std]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov  = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return projected_mean, projected_cov + innovation_cov

    def update(self, mean, covariance, measurement, confidence=0.0):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation     = measurement - projected_mean
        new_mean       = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Mahalanobis distance dùng cho MC gating."""
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj  = cov_proj[:2, :2]
            measurements = measurements[:, :2]
        chol = np.linalg.cholesky(cov_proj)
        d    = measurements - mean_proj
        z    = scipy.linalg.solve_triangular(
            chol, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


# ─────────────────────────────────────────────────────────────
# STrack
# ─────────────────────────────────────────────────────────────

class STrack(BaseTrack):
    """
    Single track với:
    - State space (x, y, a, h) theo code gốc
    - NSA Kalman Filter
    - EMA appearance feature
    - time_since_update là instance variable (reset=0 khi update, +=1 khi predict)
    """

    shared_kalman = NSAKalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):
        super().__init__()  # time_since_update = 0 (instance var)

        self._tlwh        = np.asarray(tlwh, dtype=np.float64)
        self.score        = score
        self.is_activated = False
        self.tracklet_len = 0
        self.mean         = None
        self.covariance   = None

        # EMA appearance
        self.smooth_feat = None
        self.curr_feat   = None
        self.features    = deque([], maxlen=feat_history)
        self.alpha       = 0.9   # EMA decay

        if feat is not None:
            self._update_features(feat)

    # ── EMA ─────────────────────────────────────────────────

    def _update_features(self, feat):
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-8)
        self.features.append(feat)

    # ── BBox converters ──────────────────────────────────────

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """tlwh [x1,y1,w,h] → xyah [cx,cy,a,h] — state space của Kalman gốc."""
        ret    = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2   # top-left → center
        ret[2]  /= ret[3]        # w → aspect ratio a = w/h
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah):
        ret    = np.asarray(xyah).copy()
        ret[2] *= ret[3]         # a → w
        ret[:2] -= ret[2:] / 2  # center → top-left
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        return self.xyah_to_tlwh(self.mean[:4])

    @property
    def tlbr(self):
        ret      = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret      = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    # ── Kalman predict ───────────────────────────────────────

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0   # va = 0
            mean_state[7] = 0   # vh = 0
        self.mean, self.covariance = self.shared_kalman.predict(
            mean_state, self.covariance)
        self.time_since_update += 1   # tăng sau predict — theo gốc Track.predict()

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean       = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance  for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for i, (m, c) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean              = m
            stracks[i].covariance        = c
            stracks[i].time_since_update += 1   # tăng sau predict

    # ── Lifecycle ────────────────────────────────────────────

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.mean, self.covariance = self.shared_kalman.initiate(
            self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len      = 0
        self.state             = TrackState.Tracked
        self.is_activated      = True   # woC: activate ngay (n_init=1)
        self.frame_id          = frame_id
        self.start_frame       = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh),
            confidence=new_track.score)
        if new_track.curr_feat is not None:
            self._update_features(new_track.curr_feat)
        self.tracklet_len      = 0
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.frame_id          = frame_id
        self.score             = new_track.score
        self.time_since_update = 0   # reset — theo gốc Track.update()
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        self.frame_id     = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track.tlwh),
            confidence=new_track.score)
        if new_track.curr_feat is not None:
            self._update_features(new_track.curr_feat)
        self.state             = TrackState.Tracked
        self.is_activated      = True
        self.score             = new_track.score
        self.time_since_update = 0   # reset — theo gốc Track.update()

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


# ─────────────────────────────────────────────────────────────
# Distance / Cost functions
# ─────────────────────────────────────────────────────────────

def _iou_batch(boxes1, boxes2):
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


def iou_distance(atracks, btracks):
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    return 1.0 - _iou_batch(
        np.array([t.tlbr for t in atracks], dtype=np.float32),
        np.array([t.tlbr for t in btracks], dtype=np.float32))


def embedding_distance(tracks, detections, metric='cosine'):
    """Cosine distance dựa trên EMA smooth_feat."""
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


def gate_cost_matrix(cost_matrix, tracks, detections,
                     track_indices, detection_indices,
                     lambda_mc=0.98, gated_cost=1e5):
    """
    MC (Matching with Cascade) gating — theo gốc linear_assignment.gate_cost_matrix():
    1. Set cost = gated_cost nếu Mahalanobis distance > chi2inv95[4]
    2. Fuse: cost = lambda * cost + (1-lambda) * mahal_distance
    """
    measurements = np.array([
        STrack.tlwh_to_xyah(detections[j].tlwh)
        for j in detection_indices], dtype=np.float64)

    for row, ti in enumerate(track_indices):
        track = tracks[ti]
        mahal = track.shared_kalman.gating_distance(
            track.mean, track.covariance, measurements)
        cost_matrix[row, mahal > CHI2INV95_4] = gated_cost
        # MC fuse
        cost_matrix[row] = lambda_mc * cost_matrix[row] + (1 - lambda_mc) * mahal
    return cost_matrix


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])))
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = np.array([[i, x[i]] for i in range(len(x)) if x[i] >= 0], dtype=int)
    u_a     = list(np.where(x < 0)[0])
    u_b     = list(np.where(y < 0)[0])
    return matches, u_a, u_b


def joint_stracks(lista, listb):
    exists = {t.track_id for t in lista}
    return lista + [t for t in listb if t.track_id not in exists]


def sub_stracks(lista, listb):
    remove = {t.track_id for t in listb}
    return [t for t in lista if t.track_id not in remove]


def remove_duplicate_stracks(stracksa, stracksb, iou_thresh=0.15):
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
# StrongSORT — theo đúng logic code gốc dyhBUPT
# ─────────────────────────────────────────────────────────────

class StrongSort:
    """
    StrongSORT theo code gốc dyhBUPT/StrongSORT:

    Flags implemented:
    - NSA  : scale measurement noise std theo (1 - confidence) trong project()
    - EMA  : smooth_feat = alpha * old + (1-alpha) * new, normalize
    - woC  : n_init=1, activate ngay frame đầu; cascade match tất cả cùng lúc
    - MC   : fuse embedding cost với Mahalanobis gating distance

    Association flow (theo Tracker._match() gốc):
    1. Cascade match confirmed tracks với appearance + MC gating
    2. IoU match: unconfirmed tracks + confirmed tracks có age==1
       (tracks vừa bị missed 1 frame)
    3. Tạo track mới cho detections còn lại
    """

    def __init__(
        self,
        track_high_thresh:   float = 0.6,
        new_track_thresh:    float = 0.65,
        track_buffer:        int   = 60,
        match_thresh:        float = 0.8,   # embedding matching threshold
        max_iou_distance:    float = 0.7,   # IoU second association threshold
        lambda_mc:           float = 0.98,  # MC fuse weight (embedding vs mahal)
        ema_alpha:           float = 0.9,
        max_cascade_depth:   int   = 50,
    ):
        self.track_high_thresh = track_high_thresh
        self.new_track_thresh  = new_track_thresh
        self.max_time_lost     = track_buffer
        self.match_thresh      = match_thresh
        self.max_iou_distance  = max_iou_distance
        self.lambda_mc         = lambda_mc
        self.ema_alpha         = ema_alpha
        self.max_cascade_depth = max_cascade_depth

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        BaseTrack.clear_count()

    def update(self, detections: List[Dict], frame_id: int) -> List[STrack]:
        self.frame_id = frame_id

        activated_stracks = []
        refind_stracks    = []
        lost_stracks      = []
        removed_stracks   = []

        # ── Tạo STrack từ detections ─────────────────────────
        dets = [d for d in detections if d['score'] >= self.track_high_thresh]
        det_stracks = []
        for d in dets:
            x1, y1, x2, y2 = d['tlbr']
            st = STrack([x1, y1, x2 - x1, y2 - y1], d['score'], feat=d.get('feat'))
            st.alpha = self.ema_alpha
            det_stracks.append(st)

        # ── Kalman predict ────────────────────────────────────
        strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)   # time_since_update += 1

        # ── Association 1: Cascade match (woC + MC) ──────────
        # woC: match TẤT CẢ confirmed tracks cùng lúc (không phân level)
        # MC : fuse embedding cost với Mahalanobis gating
        confirmed   = [i for i, t in enumerate(strack_pool) if t.is_activated]
        unconfirmed = [i for i, t in enumerate(strack_pool) if not t.is_activated]

        det_indices = list(range(len(det_stracks)))

        # Tính embedding cost matrix (NxM)
        tracks_conf = [strack_pool[i] for i in confirmed]
        has_feat = (
            len(tracks_conf) > 0 and
            all(t.smooth_feat is not None for t in tracks_conf) and
            all(d.curr_feat   is not None for d in det_stracks)
        )

        if has_feat:
            cost_matrix = embedding_distance(tracks_conf, det_stracks)
            # MC gating: gate + fuse với Mahalanobis
            cost_matrix = gate_cost_matrix(
                cost_matrix, strack_pool, det_stracks,
                confirmed, det_indices, lambda_mc=self.lambda_mc)
        else:
            # fallback IoU nếu không có feat
            cost_matrix = iou_distance(tracks_conf, det_stracks)

        matches_a, u_conf, u_det = linear_assignment(cost_matrix, self.match_thresh)

        for li, ld in matches_a:
            track = tracks_conf[li]
            det   = det_stracks[ld]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        # ── Association 2: IoU match ──────────────────────────
        # Theo gốc: unconfirmed + confirmed tracks với age==1
        unmatched_conf_global = [confirmed[i] for i in u_conf]
        iou_candidates = unconfirmed + [
            k for k in unmatched_conf_global
            if strack_pool[k].time_since_update == 1
        ]
        remaining_unmatched_conf = [
            k for k in unmatched_conf_global
            if strack_pool[k].time_since_update != 1
        ]

        tracks_iou  = [strack_pool[k] for k in iou_candidates]
        dets_remain = [det_stracks[j] for j in u_det]

        if tracks_iou and dets_remain:
            iou_cost = iou_distance(tracks_iou, dets_remain)
            matches_b, u_iou_t, u_iou_d = linear_assignment(
                iou_cost, self.max_iou_distance)

            matched_t2 = set(range(len(tracks_iou))) - set(u_iou_t)
            matched_d2 = set(range(len(dets_remain))) - set(u_iou_d)

            for lt, ld in matches_b:
                track = tracks_iou[lt]
                det   = dets_remain[ld]
                if track.state == TrackState.Tracked:
                    track.update(det, frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, frame_id, new_id=False)
                    refind_stracks.append(track)

            # Unmatched sau cả 2 vòng → lost
            u_iou_t_global = [iou_candidates[i] for i in u_iou_t]
            u_det_final    = [u_det[j] for j in u_iou_d]
        else:
            u_iou_t_global = iou_candidates
            u_det_final    = list(u_det)

        # ── Mark lost ─────────────────────────────────────────
        for k in remaining_unmatched_conf + u_iou_t_global:
            track = strack_pool[k]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── Tạo track mới ─────────────────────────────────────
        for j in u_det_final:
            det = det_stracks[j]
            if det.score >= self.new_track_thresh:
                det.activate(frame_id)
                activated_stracks.append(det)

        # ── Lost → removed ────────────────────────────────────
        for track in self.lost_stracks:
            if frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ── Update pools ──────────────────────────────────────
        self.tracked_stracks = [t for t in self.tracked_stracks
                                 if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]