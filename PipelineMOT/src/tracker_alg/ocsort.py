"""
OC-SORT tracker với hỗ trợ tùy chọn feature embedding.
Adopted from the SORT script by Alex Bewley (alex@bewley.ai), refactored from Deep OC-SORT.
 
Pipeline:
    - Input mỗi frame: list of dict [{'tlbr', 'score', 'feat'}, ...]
    - Output mỗi frame: list of STrack, mỗi STrack chứa tlbr và track_id
    
[TODO]: Định nghĩa hàm convert_bbox_to_z_new: biến đổi từ [x1,y1,x2,y2] sang [x,y] của pitch
        và convert_x_to_bbox_new: giống inverse_homography biến đổi từ [x,y] của pitch sang [x1,y1,x2,y2]
"""

import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

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
    """
    Tính ma trận IoU giữa hai tập bounding boxes.
 
    Args:
        bboxes1 (np.array): Shape (M, 4), định dạng [x1, y1, x2, y2].
        bboxes2 (np.array): Shape (N, 4), định dạng [x1, y1, x2, y2].
 
    Returns:
        np.array: Ma trận IoU shape (M, N). Giá trị trong [0, 1].
    """
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
    """
    Giải bài toán Hungarian assignment để tìm matching tối ưu (minimize cost).
    Ưu tiên dùng `lap.lapjv` nếu có (nhanh hơn), fallback về `scipy`.
 
    Args:
        cost_matrix (np.array): Ma trận chi phí shape (M, N).
 
    Returns:
        np.array: Shape (K, 2) — mỗi hàng là [det_idx, trk_idx] đã được match.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    """
    Tính Adaptive Weighting (AW) cho embedding cost.
    Weight của embedding được scale down nếu 2 candidate tốt nhất quá gần nhau
    (tức embedding không đủ discriminative để phân biệt).
    Code implement khác paper, dùng chia thay hiệu
 
    Args:
        emb_cost         (np.array): Ma trận embedding similarity shape (M, N).
        w_association_emb   (float): Weight ban đầu cho embedding.
        bottom              (float): Cut off weight, càng cao thì trọng số cho emb càng lớn
 
    Returns:
        np.array: Ma trận embedding cost đã được adaptive-weight, shape (M, N).
    """
    w_emb = np.full_like(emb_cost, w_association_emb)   #fill full aw

    for idx in range(emb_cost.shape[0]):
        inds = np.argsort(-emb_cost[idx])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if emb_cost[idx, inds[0]] == 0:
            row_weight = 0
        else:
            row_weight = 1 - max((emb_cost[idx, inds[1]] / emb_cost[idx, inds[0]]) - bottom, 0) / (1 - bottom)
        w_emb[idx] *= row_weight

    for idj in range(emb_cost.shape[1]):
        inds = np.argsort(-emb_cost[:, idj])
        # If there's less than two matches, just keep original weight
        if len(inds) < 2:
            continue
        if emb_cost[inds[0], idj] == 0:
            col_weight = 0
        else:
            col_weight = 1 - max((emb_cost[inds[1], idj] / emb_cost[inds[0], idj]) - bottom, 0) / (1 - bottom)
        w_emb[:, idj] *= col_weight

    return w_emb * emb_cost

def associate_euclidean(
    detections, trackers, distance_threshold, velocities, previous_obs, vdc_weight, emb_cost, w_assoc_emb, aw_off, aw_param
):
    """
    Association Round 1 dùng Euclidean distance thay IoU — cho trường hợp new_kf=True.
    Returns:
        matches             (np.array): Shape (K, 2) — [det_idx, trk_idx].
        unmatched_detections(np.array): Indices detections không được match.
        unmatched_trackers  (np.array): Indices trackers không được match.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # Tính velocity direction dùng bottom center
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    # Euclidean distance thay IoU
    # dets: bottom center = ((x1+x2)/2, y2)
    # trks: x,y từ KF predict = (trks[:,0], trks[:,1])
    det_centers = np.stack([
        (detections[:, 0] + detections[:, 2]) / 2.0,  # cx
        detections[:, 3]                               # y2 = bottom
    ], axis=1)  # (M, 2)
    trk_centers = trackers[:, :2]  # (N, 2) — x,y từ [x,y,x,y,0]

    from scipy.spatial.distance import cdist
    dist_matrix = cdist(det_centers, trk_centers, metric='euclidean')  # (M, N) >=0

    # dùng negative distance để Hungarian minimize → maximize similarity
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(dist_matrix.shape) > 0:
        a = (dist_matrix < distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost[dist_matrix >= distance_threshold] = 0  # gating
                if not aw_off:
                    emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost *= w_assoc_emb

            # đổi về similarity để nhất quán logic optimize trong hungarian matching
            similarity_matrix = 1 - np.clip(dist_matrix / distance_threshold, 0, 1)  # [0,1]
            final_cost = -(similarity_matrix + angle_diff_cost + emb_cost)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched với distance quá lớn
    matches = []
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] >= distance_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate(
    detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight, emb_cost, w_assoc_emb, aw_off, aw_param
):
    """
    Association chính (Round 1 — ORM) kết hợp IoU + velocity direction + embedding.
 
    Args:
        detections   (np.array): Shape (M, 5) — [x1, y1, x2, y2, score].
        trackers     (np.array): Shape (N, 5) — predicted boxes của các trackers.
        iou_threshold   (float): Ngưỡng IoU tối thiểu để chấp nhận match.
        velocities   (np.array): Shape (N, 2) — vector vận tốc (dy, dx) của từng tracker.
        previous_obs (np.array): Shape (N, 5) — observation gần nhất của từng tracker,
                                 dùng để tính hướng di chuyển kỳ vọng.
        vdc_weight      (float): Weight của velocity direction consistency (OCM inertia).
        emb_cost     (np.array|None): Ma trận embedding similarity shape (M, N),
                                      hoặc None nếu không dùng embedding.
        w_assoc_emb     (float): Weight ban đầu cho embedding cost.
        aw_off           (bool): Nếu True, tắt adaptive weighting, dùng fixed weight.
        aw_param        (float): Tham số `bottom` cho adaptive weighting.
 
    Returns:
        matches             (np.array): Shape (K, 2) — [det_idx, trk_idx].
        unmatched_detections(np.array): Indices detections không được match.
        unmatched_trackers  (np.array): Indices trackers không được match.
    """
    
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )
        
    # Tính velocity direction giữa detection và previous observation của tracker
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    # Normalize về [0, 1]: góc càng nhỏ (cùng hướng) → giá trị càng cao
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # Mỗi det/trk chỉ có 1 match khả dĩ → gán trực tiếp, không cần solver
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Kết hợp IoU + angle + embedding thành final cost rồi giải Hungarian
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost *= w_assoc_emb

            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def speed_direction_batch(dets, tracks):
    """
    Tính vector hướng di chuyển chuẩn hóa từ mỗi tracker đến mỗi detection.
    Dùng để đánh giá velocity direction consistency trong OCM.
 
    Args:
        dets   (np.array): Shape (M, 4+) — bounding boxes của detections.
        tracks (np.array): Shape (N, 4+) — previous observations của trackers.
 
    Returns:
        dy (np.array): Shape (N, M) — thành phần hướng dọc (đã chuẩn hóa).
        dx (np.array): Shape (N, M) — thành phần hướng ngang (đã chuẩn hóa).
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx 


# =============================================================================
# Utility functions cho KalmanBoxTracker
# =============================================================================

def k_previous_obs(observations, cur_age, k):
    """
    Lấy observation gần nhất trong k frame trước của một tracker.
    Dùng để tính velocity direction trong OCM.
 
    Args:
        observations (dict): {frame_age: bbox} — lịch sử observation của tracker.
        cur_age       (int): Age hiện tại của tracker (số frame kể từ khi tạo).
        k             (int): Số frame nhìn lại tối đa (= delta_t).
 
    Returns:
        list: Bbox [x1, y1, x2, y2, score] của observation tìm được,
              hoặc [-1, -1, -1, -1, -1] nếu không có observation nào.
    """
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def convert_bbox_to_z(bbox):
    """
    Chuyển bbox [x1, y1, x2, y2] sang state vector [x, y, s, r] khi new_kf = False.
    với x, y là tâm box, s là w*h, r là w/h
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    """
    Chuyển state vector [x_center, y_center, s, r] sang bbox [x1, y1, x2, y2] khi new_kf = False
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))

def convert_bbox_to_z_new(bbox):
    """
    Chuyển bbox [x1, y1, x2, y2] sang state vector [x, y] khi new_kf = True.
    """
    # import từ pitch localization
    return np.array([0,0]).reshape((2, 1)) # toy data

def convert_x_to_bbox_new(x):
    """
    Chuyển state vector [x, y] sang cx_bot, cy_bot new_kf = True
    """
    #cx_bot, cy_bot = convert_xy(x)     HÀM BIẾN ĐỔI TỪ X,Y TRÊN SÂN -> X, Y Ở GIỮA PHÍA DƯỚI TRÊN ẢNH
    cx_bot, cy_bot = 5, 5 #placeholder
    
    return cx_bot, cy_bot



# =============================================================================
# KalmanBoxTracker
# =============================================================================

class KalmanBoxTracker(object):
    """
    Represents the internal state of an individually tracked object observed as a bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, emb=None, alpha=0, new_kf=False):
        """
        Initialises a tracker using initial bounding box.
        new_kf: false -> state dạng [x,y,s,r] trên trục hình ảnh
                true -> state dạng [x,y] trên tọa độ pitch
        """
        from .kalman_filter import KalmanFilterNew as KalmanFilter
        self.new_kf = new_kf
        
        if not new_kf:  # trường hợp false
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    #x  y  s  r  x' y' s'
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            
            self.kf.R[2:, 2:] *= 10.    # tăng noise cho s, r
            self.kf.P[4:, 4:] *= 1000.  # tăng sự không chắc chắn cho velocity tức x', y', s'
            self.kf.P *= 10.            # sự không chắc chắn toàn bộ
            self.kf.Q[-1, -1] *= 0.01   # s' thường thay đổi ít
            self.kf.Q[4:, 4:] *= 0.01   # giả định s' thay đổi chậm

            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox
            
            self.kf.x[:4] = self.bbox_to_z_func(bbox)
            
        else:   #new_kf = True -> point [x, y, vx, vy]
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = np.array(
                [
                    #x  y  x' y'
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]
                ]
            )
            # self.kf.R = self.kf.R     #không đo kích thước -> noise đều
            self.kf.P[2:, 2:] *= 100.0  # high uncertainty for unobservable initial velocities
            self.kf.P *= 10.0
            self.kf.Q[2:, 2:] *= 0.01   # vx, vy
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
            
            self.kf.x[:2] = self.bbox_to_z_func(bbox)
            self.last_wh = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])  # x2-x1, y2-y1
        
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
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
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            self.frozen = False

            if self.last_observation.sum() >= 0:  # has previous observation
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
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if not self.new_kf:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        if self.new_kf:
            cx, cy_bottom = self.x_to_bbox_func(self.kf.x[:2])  # tọa độ sân → bottom center pixel
            w, h = self.last_wh
            return np.array([[cx - w/2, cy_bottom - h, cx + w/2, cy_bottom]])
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


# =============================================================================
# OCSort
# =============================================================================

class OCSort(object):
    def __init__(
        self,
        use_project = False,
        det_thresh = 0.3,
        use_emb = False,
        max_age=30,
        min_hits=5,
        iou_threshold= 0.4,
        distance_threshold = 100,
        delta_t=3,
        inertia=0.2,    #lamda trong cost(iou) + lamda*cost(velocity)
        w_association_emb=0.8, # weight embed khởi tạo (a_w), nếu không dùng AW, tính cost(ocsort) + w_association_emb*emb
        alpha_fixed_emb=0.95,   #fixed alpha trong EMA, càng lớn càng ưu tiên lịch sử
        aw_param=0.5,   #epsilon, càng cao càng dễ dãi với tỉ lệ top2/top1 -> weight càng có thể lớn
        aw_off=False
    ):
        """
        Args:
            use_project      (bool):  True nếu muốn sử dụng tọa độ chiếu x,y làm observation.
            det_thresh       (float): Ngưỡng confidence tối thiểu của detection đầu vào.
                                      Detections có score <= det_thresh bị loại bỏ.
            use_emb           (bool): Có dùng feature embedding trong association hay không.
                                      False → chỉ dùng IoU + velocity direction.
            max_age            (int): Số frame tối đa một tracker được giữ khi không match.
                                      Sau max_age frame miss, tracker bị xóa.
            min_hits           (int): Số frame match liên tiếp tối thiểu để một track
                                      được output ra ngoài (confirmed track).
            iou_threshold    (float): Ngưỡng IoU tối thiểu để chấp nhận một match
                                      trong cả 2 round association.
            distance         (float): Ngưỡng khoảng cách euclid  tối đa để chấp nhận một match (theo pixel)
                                      trong cả 2 round association.
            delta_t            (int): Số frame nhìn lại để tính velocity cho OCM.
                                      Càng lớn → velocity ổn định hơn nhưng lag hơn.
            inertia          (float): Weight của velocity direction cost trong Round 1 (vdc_weight).
                                      Càng cao → ưu tiên tracker đi đúng hướng dự đoán.
            w_association_emb(float): Weight ban đầu cho embedding cost trong association. Nếu không dùng AW thì weight giữ nguyên.
                                      Ngược lại, sẽ giảm dần nếu embedding không đủ phân biệt.
            alpha_fixed_emb  (float): Hệ số EMA cố định khi update embedding.
                                      Giá trị cao → giữ embedding lịch sử nhiều hơn.
            aw_param         (float): Tham số `bottom` của Adaptive Weighting.
                                      Điều chỉnh mức độ giảm weight khi embedding không discriminative.
            aw_off            (bool): Nếu True, tắt Adaptive Weighting, dùng w_association_emb cố định.
        """
        self.max_age = max_age      # số frame lost tối đa
        self.min_hits = min_hits    # Số frame liên tiếp phải match để được confirmed
        self.iou_threshold = iou_threshold  # ngưỡng iou trong association
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh    # ngưỡng lọc det đầu vào
        self.delta_t = delta_t          # OCM velocity
        self.asso_func = iou_batch
        self.inertia = inertia          # OCM quán tính
        self.w_association_emb = w_association_emb  # aw + wb
        self.alpha_fixed_emb = alpha_fixed_emb  # fixed alpha trong smooth embed
        self.aw_param = aw_param        #epsilon
        self.use_emb = use_emb
        KalmanBoxTracker.count = 0
        
        self.aw_off = aw_off
        self.new_kf = use_project
        

    def update(self, detections, frame_id):
        """
        Params:
            detections: list of dict, each with keys:
                - 'tlbr'  : [x1, y1, x2, y2]
                - 'score' : float
                - 'feat'  : np.array (embedding vector)
        Requires: called once per frame, even with empty list.
        Returns: np.array of shape (N, 5) — [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        if len(detections) == 0:
            for trk in self.trackers:
                trk.predict()
                trk.update(None)

            # Xóa tracker chết
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            return np.empty((0, 5))
    
        # Parse input
        dets = np.array([[*d['tlbr'], d['score']] for d in detections])   # (N, 5)
        dets_embs = np.array([d['feat'] for d in detections]) if self.use_emb else None            # (N, D)

        # Filter by detection threshold
        remain_inds = dets[:, 4] > self.det_thresh
        dets = dets[remain_inds]
        if self.use_emb:
            dets_embs = dets_embs[remain_inds]

        if len(dets) == 0:
            # Vẫn aging trackers
            for trk in self.trackers:
                trk.predict()
                trk.update(None)
            # Xóa tracker chết (code giống phần trên)
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            return np.empty((0, 5))
    
        if self.use_emb:
            # Adaptive embedding weight based on detection confidence
            trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
            af = self.alpha_fixed_emb
            # From [alpha_fixed_emb, 1], goes to 1 as detector is less confident
            dets_alpha = af + (1 - af) * (1 - trust)    #alpha_t

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0] 
            if not self.new_kf:
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            else:   # pad thêm để giữ shape, thực tế chỉ cần pos[0] và pos[1] tức x,y để tính euclid distance
                trk[:] = [pos[0], pos[1], pos[0], pos[1], 0] 
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                if self.use_emb:
                    trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        if self.use_emb:
            trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers]
        )

        """
            First round of association - ORM
        """
        # (M detections X N tracks, final score)
        if self.use_emb and dets.shape[0] > 0 and trk_embs.shape[0] > 0:
            stage1_emb_cost = dets_embs @ trk_embs.T
        else:
            stage1_emb_cost = None

        if not self.new_kf:
            matched, unmatched_dets, unmatched_trks = associate(
                dets,
                trks,
                self.iou_threshold,
                velocities,
                k_observations,
                self.inertia,
                stage1_emb_cost,
                self.w_association_emb,
                self.aw_off,
                self.aw_param,
            )
        else:   #associaate theo khoảng cách Euclid
            matched, unmatched_dets, unmatched_trks = associate_euclidean(
                dets,                  # [x1,y1,x2,y2,score] — lấy bottom center bên trong
                trks,                  # [x,y,x,y,0] — lấy x,y bên trong
                self.distance_threshold,    
                velocities,
                k_observations,
                self.inertia,
                stage1_emb_cost,
                self.w_association_emb,
                self.aw_off,
                self.aw_param,
            )
            
        for m in matched:   #[[det, track],...]
            self.trackers[m[1]].update(dets[m[0], :])
            if self.use_emb:
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        """
            Second round of association by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]  # last_observation vẫn là [x1,y1,x2,y2,score] không cần sửa association round2
            
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
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity.
                """
                rematched_indices = linear_assignment(rematching_cost)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    if self.use_emb:
                        self.trackers[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                emb= dets_embs[i] if self.use_emb else None,
                alpha= dets_alpha[i] if self.use_emb else 0,
                new_kf= self.new_kf,
            )
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive IDs
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            arr = np.concatenate(ret)          # (N, 5)
            return [STrack(row) for row in arr]
        return []