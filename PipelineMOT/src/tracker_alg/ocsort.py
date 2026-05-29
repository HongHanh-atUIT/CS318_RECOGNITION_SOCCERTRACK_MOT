"""
OC-SORT tracker với hỗ trợ tùy chọn feature embedding.
Adopted from the SORT script by Alex Bewley (alex@bewley.ai), refactored from Deep OC-SORT.

Pipeline:
    - Input mỗi frame: list of dict [{'tlbr', 'score', 'feat'}, ...]
    - Output mỗi frame: list of STrack, mỗi STrack chứa tlbr và track_id

use_project=True:
    - KalmanBoxTracker dùng state [x, y, vx, vy] trong tọa độ pitch thay vì [x,y,s,r] pixel
    - bbox_to_z_func: [x1,y1,x2,y2] → [x,y] pitch qua homography  (closure capture H)
    - x_to_bbox_func: [x,y] pitch → bottom-center pixel qua H_inv  (closure capture H_inv)
    - associate_euclidean thay associate (round 1)
    - Cần truyền H, H_inv, frame_w, frame_h khi khởi tạo OCSort
"""

import numpy as np
from typing import Optional
from scipy.optimize import linear_sum_assignment
from src.pitch_localization import box_to_pitch_xy, pitch_xy_to_bottom_center

# =============================================================================
# STrack — Wrapper output để tương thích với pipeline
# =============================================================================

class STrack:
    """Wrapper nhỏ để OCSort output tương thích với pipeline."""
    def __init__(self, row):
        # row: [x1, y1, x2, y2, track_id]
        self.tlbr = np.array(row[:4])
        self.track_id = int(row[4])

# =============================================================================
# Association functions
# =============================================================================

def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    for idx in range(emb_cost.shape[0]):
        inds = np.argsort(-emb_cost[idx])
        if len(inds) < 2:
            continue
        if emb_cost[idx, inds[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max(
                (emb_cost[idx, inds[1]] / emb_cost[idx, inds[0]]) - bottom, 0
            ) / (1 - bottom)
        w_emb[idx] *= row_weight
    for idj in range(emb_cost.shape[1]):
        inds = np.argsort(-emb_cost[:, idj])
        if len(inds) < 2:
            continue
        if emb_cost[inds[0], idj] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max(
                (emb_cost[inds[1], idj] / emb_cost[inds[0], idj]) - bottom, 0
            ) / (1 - bottom)
        w_emb[:, idj] *= col_weight
    return w_emb * emb_cost


def associate_euclidean(
    detections, trackers, distance_threshold, velocities,
    previous_obs, vdc_weight, emb_cost, w_assoc_emb, aw_off, aw_param
):
    """
    Round 1 dùng Euclidean distance — cho new_kf=True (use_project).
    trackers[:,0:2] là bottom-center pixel từ x_to_bbox_func (sau H_inv).
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, -1, 1)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    # det bottom-center pixel
    det_centers = np.stack([
        (detections[:, 0] + detections[:, 2]) / 2.0,
        detections[:, 3]
    ], axis=1)  # (M, 2)

    # trk: đã là bottom-center pixel được lưu vào trks[:,0:2]
    trk_centers = trackers[:, :2]  # (N, 2)

    from scipy.spatial.distance import cdist
    dist_matrix = cdist(det_centers, trk_centers, metric='euclidean')  # (M, N)

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T * scores

    if min(dist_matrix.shape) > 0:
        a = (dist_matrix < distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost_contrib = 0
            else:
                emb_cost[dist_matrix >= distance_threshold] = 0
                if not aw_off:
                    emb_cost_contrib = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost_contrib = emb_cost * w_assoc_emb

            similarity_matrix = 1 - np.clip(dist_matrix / distance_threshold, 0, 1)
            final_cost = -(similarity_matrix + angle_diff_cost + emb_cost_contrib)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections, unmatched_trackers = [], []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] >= distance_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate(
    detections, trackers, iou_threshold, velocities,
    previous_obs, vdc_weight, emb_cost, w_assoc_emb, aw_off, aw_param
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, -1, 1)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost_contrib = 0
            else:
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost_contrib = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost_contrib = emb_cost * w_assoc_emb
            final_cost = -(iou_matrix + angle_diff_cost + emb_cost_contrib)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections, unmatched_trackers = [], []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


# =============================================================================
# Utility functions
# =============================================================================

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    return observations[max(observations.keys())]


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] → [x,y,s,r] — dùng khi new_kf=False."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    """[x_c,y_c,s,r] → [x1,y1,x2,y2] — dùng khi new_kf=False."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))


# =============================================================================
# KalmanBoxTracker
# =============================================================================

class KalmanBoxTracker(object):
    """
    Đại diện internal state của một tracked object.

    new_kf=False : state [x, y, s, r, vx, vy, vs] — pixel, dim_x=7, dim_z=4
    new_kf=True  : state [x, y, vx, vy]            — pitch (m), dim_x=4, dim_z=2
                   Cần H, H_inv, frame_w, frame_h để tạo closure bbox↔pitch.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, emb=None, alpha=0,
                 new_kf=False,
                 H=None, H_inv=None, frame_w=1920, frame_h=1080):
        from .kalman_filter import KalmanFilterNew as KalmanFilter
        self.new_kf = new_kf

        if not new_kf:
            # ── State [x, y, s, r, vx, vy, vs] ─────────────────────
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array([
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ])
            self.kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ])
            self.kf.R[2:, 2:] *= 10.
            self.kf.P[4:, 4:] *= 1000.
            self.kf.P *= 10.
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01

            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox
            self.kf.x[:4] = self.bbox_to_z_func(bbox)

        else:
            # ── State [x, y, vx, vy] trong tọa độ pitch ─────────────
            # FIX: tạo closure capture H, H_inv, frame_w, frame_h
            # để không cần global variables
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            self.kf.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ])
            self.kf.P[2:, 2:] *= 100.0
            self.kf.P *= 10.0
            self.kf.Q[2:, 2:] *= 0.01

            # Lưu để dùng trong last_wh và get_state
            self.last_wh = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])

            # ── Closures ─────────────────────────────────────────────
            _H, _H_inv, _fw, _fh = H, H_inv, frame_w, frame_h

            def _bbox_to_z_pitch(bbox):
                """[x1,y1,x2,y2,...] → [x,y] pitch, shape (2,1)."""
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                tlwh = np.array([bbox[0], bbox[1], w, h])
                xy = box_to_pitch_xy(tlwh, H=_H, frame_w=_fw, frame_h=_fh)
                return xy.reshape((2, 1))

            def _x_to_bc_pixel(x):
                """
                State vector (bất kỳ độ dài) → bottom-center pixel, shape (1,2).
                Chỉ dùng 2 phần tử đầu (x, y pitch).
                Trả về shape (1,2) để:
                  - predict(): history[-1][0] = [bx, by] ✓
                  - get_state(): bc[0] = [bx, by]       ✓
                """
                xy = x[:2].flatten()
                bx, by = pitch_xy_to_bottom_center(xy, H_inv=_H_inv,
                                                   frame_w=_fw, frame_h=_fh)
                return np.array([[float(bx), float(by)]])  # shape (1,2)

            self.bbox_to_z_func = _bbox_to_z_pitch
            self.x_to_bbox_func = _x_to_bc_pixel
            self.kf.x[:2] = self.bbox_to_z_func(bbox)

        # ── Common state ──────────────────────────────────────────────
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history_observations = []
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.emb = emb
        self.frozen = False

    def update(self, bbox):
        if bbox is not None:
            self.frozen = False
            if self.last_observation.sum() >= 0:
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            if self.new_kf:
                self.last_wh = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(bbox)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def predict(self):
        if not self.new_kf:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # FIX: x_to_bbox_func trả về shape (1,4) cho new_kf=False
        #                                    (1,2) cho new_kf=True
        # predict()[0] sẽ là [x1,y1,x2,y2] hoặc [bx,by] tương ứng
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Trả về bbox pixel shape (1,4) = [x1,y1,x2,y2].

        new_kf=False: dùng x_to_bbox_func trực tiếp (trả về (1,4)).
        new_kf=True : lấy bottom-center pixel từ pitch state + last_wh.
        """
        if self.new_kf:
            # FIX: x_to_bbox_func trả về (1,2), không unpack trực tiếp
            bc = self.x_to_bbox_func(self.kf.x[:2])  # shape (1,2)
            bx, by = float(bc[0, 0]), float(bc[0, 1])
            w, h = self.last_wh
            return np.array([[bx - w/2, by - h, bx + w/2, by]])
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


# =============================================================================
# OCSort
# =============================================================================

class OCSort(object):
    def __init__(
        self,
        use_project=False,
        # ── Pitch projection ──────────────────────────────────────────
        H: Optional[np.ndarray] = None,
        H_inv: Optional[np.ndarray] = None,
        frame_w: int = 1920,
        frame_h: int = 1080,
        # ── Standard params ──────────────────────────────────────────
        det_thresh=0.3,
        use_emb=False,
        max_age=30,
        min_hits=5,
        iou_threshold=0.4,
        distance_threshold=100,
        delta_t=3,
        inertia=0.2,
        w_association_emb=0.8,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        aw_off=False,
    ):
        """
        Args:
            use_project (bool): True → dùng tọa độ pitch (x,y) làm KF state.
                                Cần truyền H và H_inv.
            H           : Homography matrix pixel→pitch, shape (3,3).
                          Lấy từ pitch_localization.load_homography().
            H_inv       : Inverse homography pitch→pixel, shape (3,3).
            frame_w/h   : Kích thước frame. Dùng khi H=None (normalize mode).
                          Wide-view video: frame_w=6500, frame_h=1000.
            distance_threshold (float): Ngưỡng Euclidean pixel cho associate_euclidean
                          khi use_project=True (đơn vị pixel).
        """
        # ── Pitch projection ─────────────────────────────────────────
        self.new_kf = use_project
        self.H = H
        self.H_inv = H_inv
        self.frame_w = frame_w
        self.frame_h = frame_h

        # ── Standard ─────────────────────────────────────────────────
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = iou_batch
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.use_emb = use_emb
        self.aw_off = aw_off
        KalmanBoxTracker.count = 0

    def _make_tracker(self, bbox, emb=None, alpha=0) -> KalmanBoxTracker:
        """Tạo KalmanBoxTracker với đầy đủ params pitch (nếu use_project)."""
        return KalmanBoxTracker(
            bbox,
            delta_t=self.delta_t,
            emb=emb,
            alpha=alpha,
            new_kf=self.new_kf,
            H=self.H,
            H_inv=self.H_inv,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )

    def update(self, detections, frame_id):
        """
        detections: list of dict, each with keys:
            - 'tlbr'  : [x1, y1, x2, y2]
            - 'score' : float
            - 'feat'  : np.array (embedding vector)
        Returns: list of STrack
        """
        self.frame_count += 1

        if len(detections) == 0:
            for trk in self.trackers:
                trk.predict()
                trk.update(None)
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            return np.empty((0, 5))

        dets = np.array([[*d['tlbr'], d['score']] for d in detections])
        dets_embs = np.array([d['feat'] for d in detections]) if self.use_emb else None

        remain_inds = dets[:, 4] > self.det_thresh
        dets = dets[remain_inds]
        if self.use_emb:
            dets_embs = dets_embs[remain_inds]

        if len(dets) == 0:
            for trk in self.trackers:
                trk.predict()
                trk.update(None)
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            return np.empty((0, 5))

        if self.use_emb:
            trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
            af = self.alpha_fixed_emb
            dets_alpha = af + (1 - af) * (1 - trust)

        # ── Predict tất cả trackers ───────────────────────────────────
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            if not self.new_kf:
                # pos: [x1, y1, x2, y2] shape (4,)
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            else:
                # FIX: pos = [bx, by] bottom-center pixel, shape (2,)
                # Pad thành [bx,by,bx,by,0] để giữ shape (5,)
                # associate_euclidean dùng trackers[:,0:2] = [bx,by]
                trk[:] = [pos[0], pos[1], pos[0], pos[1], 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)
            elif self.use_emb:
                trk_embs.append(self.trackers[t].get_emb())

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        if self.use_emb:
            trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0))
             for trk in self.trackers]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t)
             for trk in self.trackers]
        )

        # ── Round 1: ORM ──────────────────────────────────────────────
        if self.use_emb and dets.shape[0] > 0 and len(trk_embs) > 0:
            stage1_emb_cost = dets_embs @ trk_embs.T
        else:
            stage1_emb_cost = None

        if not self.new_kf:
            matched, unmatched_dets, unmatched_trks = associate(
                dets, trks, self.iou_threshold,
                velocities, k_observations, self.inertia,
                stage1_emb_cost, self.w_association_emb,
                self.aw_off, self.aw_param,
            )
        else:
            # FIX: dùng associate_euclidean khi use_project=True
            matched, unmatched_dets, unmatched_trks = associate_euclidean(
                dets, trks, self.distance_threshold,
                velocities, k_observations, self.inertia,
                stage1_emb_cost, self.w_association_emb,
                self.aw_off, self.aw_param,
            )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            if self.use_emb:
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        # ── Round 2: OCR (dùng last_observation — vẫn IoU) ───────────
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)

            if self.use_emb:
                left_dets_embs = dets_embs[unmatched_dets]
                left_trks_embs = trk_embs[unmatched_trks]
                emb_cost_left = left_dets_embs @ left_trks_embs.T
                rematching_cost = -(iou_left + emb_cost_left)
            else:
                rematching_cost = -iou_left

            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(rematching_cost)
                to_remove_det, to_remove_trk = [], []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    if self.use_emb:
                        self.trackers[trk_ind].update_emb(
                            dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det.append(det_ind)
                    to_remove_trk.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # ── Tạo tracker mới ───────────────────────────────────────────
        for i in unmatched_dets:
            # FIX: _make_tracker truyền H, H_inv, frame_w, frame_h
            trk = self._make_tracker(
                dets[i, :],
                emb=dets_embs[i] if self.use_emb else None,
                alpha=dets_alpha[i] if self.use_emb else 0,
            )
            self.trackers.append(trk)

        # ── Output confirmed tracks ───────────────────────────────────
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            arr = np.concatenate(ret)
            return [STrack(row) for row in arr]
        return []