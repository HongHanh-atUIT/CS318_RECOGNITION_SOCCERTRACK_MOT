"""
DeepEIoU Tracker
Điều chỉnh từ repo gốc của tác giả, fit với BaseTrack + KalmanFilter của bạn.
Input: list of dict [{'tlbr': [x1,y1,x2,y2], 'score': float, 'feat': np.array}, ...]
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import lap
from scipy.spatial.distance import cdist
from .basetrack import BaseTrack, TrackState
from collections import deque

# ------- STrack class -----------------------------------------------
class STrack(BaseTrack):
    """
    Đại diện cho một tracklet (một đối tượng đang được theo dõi).
    Không dùng Kalman Filter — vị trí bbox được cập nhật trực tiếp từ detection.
    Quản lý: state, is_activated, appearance features (EMA smooth), lịch sử bbox.
    """

    def __init__(self, tlwh, score, feat=None, feat_history=30):
        """
        Args:
            tlwh: bbox dạng [x1, y1, w, h] từ detector
            score: confidence score
            feat: appearance feature vector (optional)
            feat_history: số feature tối đa lưu trong history
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float32)  # bbox gốc từ detector
        self.is_activated = False

        self.last_tlwh = self._tlwh.copy()  # bbox frame trước, dùng cho EIoU

        self.score = score
        self.tracklet_len = 0  # số frame được tracked liên tục

        # --- Appearance feature ---
        self.smooth_feat = None   # EMA feature, dùng để so sánh appearance
        self.curr_feat = None     # feature của detection hiện tại (chưa smooth)
        self.features = deque([], maxlen=feat_history)  # lịch sử feature có giới hạn
        self.times = []           # frame_id tương ứng với từng feature được lưu
        self.alpha = 0.9          # hệ số EMA: 0.9 → ưu tiên lịch sử hơn feature mới

        if feat is not None:
            self.update_features(feat)

    # ------------------------------------------------------------------
    # Feature management
    # ------------------------------------------------------------------

    def update_features(self, feat):
        """Cập nhật curr_feat và smooth_feat bằng EMA."""
        feat /= np.linalg.norm(feat)        # normalize về unit vector
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat          # lần đầu: dùng thẳng, chưa có lịch sử
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  # normalize lại sau EMA

    # ------------------------------------------------------------------
    # Lifecycle: activate / re_activate / update
    # ------------------------------------------------------------------

    def activate(self, frame_id):
        """
        Khởi tạo tracklet mới.
        Cấp track_id mới, đặt state = Tracked.
        Chỉ đánh dấu is_activated = True ngay nếu là frame đầu tiên,
        còn không thì chờ frame tiếp theo confirm (tránh FP).
        """
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Kích hoạt lại track đã lost khi khớp với detection mới.
        Reset tracklet_len vì bị gián đoạn.
        new_id=True khi muốn cấp ID mới (tránh ID switch sau lost dài).
        """
        # Cập nhật vị trí trực tiếp từ detection mới
        self.last_tlwh = self._tlwh.copy()     # ← giữ bbox cũ, không phải bbox mới
        self._tlwh = new_track.tlwh.copy()

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)

        self.tracklet_len = 0   # reset vì đã bị gián đoạn
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        """
        Cập nhật track được matched với detection trong frame hiện tại.
        Cập nhật trực tiếp bbox, feature, score — không dùng Kalman Filter.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Lưu bbox hiện tại vào last_tlwh trước khi ghi đè
        self.last_tlwh = self._tlwh.copy()
        self._tlwh = new_track.tlwh.copy()  # cập nhật vị trí trực tiếp từ detection

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
            self.times.append(frame_id)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    # ------------------------------------------------------------------
    # BBox format converters (properties)
    # ------------------------------------------------------------------

    @property
    def tlwh(self):
        """Trả về bbox hiện tại dạng [x1, y1, w, h]."""
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Trả về bbox hiện tại dạng [x1, y1, x2, y2]."""
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]   # x2 = x1+w, y2 = y1+h
        return ret

    @property
    def last_tlbr(self):
        """Trả về bbox frame trước dạng [x1, y1, x2, y2], dùng cho EIoU."""
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Trả về bbox hiện tại dạng [cx, cy, w, h] (center-based)."""
        ret = self._tlwh.copy()
        ret[:2] += ret[2:] / 2.0   # cx = x1 + w/2, cy = y1 + h/2
        return ret

    # ------------------------------------------------------------------
    # BBox format converters (static)
    # ------------------------------------------------------------------

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """[x1, y1, w, h] → [cx, cy, w, h]."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """[x1, y1, w, h] → [cx, cy, aspect_ratio, h]. Dùng cho DeepSORT nếu cần."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]   # aspect ratio = w / h
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """[x1, y1, x2, y2] → [x1, y1, w, h]."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]   # w = x2-x1, h = y2-y1
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """[x1, y1, w, h] → [x1, y1, x2, y2]."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
    
    
# -------- Helper function --------------------------------------------
def expand(tlbr: np.ndarray, e: float) -> np.ndarray:
    """Mở rộng box theo tỷ lệ e (theo paper Deep-EIoU)."""
    x1, y1, x2, y2 = tlbr  # đúng convention STrack
    w = x2 - x1
    h = y2 - y1

    expand_w = w * (1 + 2 * e)
    expand_h = h * (1 + 2 * e)

    return np.array([
        x1 - expand_w / 2,
        y1 - expand_h / 2,
        x2 + expand_w / 2,
        y2 + expand_h / 2
    ], dtype=np.float32)
    
    
def eious(atlbrs: List[np.ndarray], btlbrs: List[np.ndarray], e: float) -> np.ndarray:
    """Tính EIoU sau khi expand."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    atlbrs_exp = np.stack([expand(tlbr, e) for tlbr in atlbrs])
    btlbrs_exp = np.stack([expand(tlbr, e) for tlbr in btlbrs])

    return _iou_batch(atlbrs_exp, btlbrs_exp)

def eiou_distance(atracks: List[STrack], btracks: List[STrack], expand: float = 0.7) -> np.ndarray:
    """Tính cost matrix dựa trên EIoU (dùng last_tlbr)."""
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    atlbrs = [t.last_tlbr for t in atracks]
    btlbrs = [t.last_tlbr for t in btracks]

    ious = eious(atlbrs, btlbrs, expand)
    return 1.0 - ious


def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Linear assignment dùng lapjv."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    return np.asarray(matches), unmatched_a, unmatched_b


def _iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    M, N = len(boxes1), len(boxes2)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    ix1 = np.maximum(boxes1[:, 0, None], boxes2[None, :, 0])
    iy1 = np.maximum(boxes1[:, 1, None], boxes2[None, :, 1])
    ix2 = np.minimum(boxes1[:, 2, None], boxes2[None, :, 2])
    iy2 = np.minimum(boxes1[:, 3, None], boxes2[None, :, 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-8
    return inter / union

def joint_stracks(lista, listb):
    """Gộp 2 list track, bỏ các track id trùng lặp"""
    exists = {t.track_id: True for t in lista}
    return lista + [t for t in listb if t.track_id not in exists]
 
 
def sub_stracks(lista, listb):
    """
    lista trừ đi các track có track_id xuất hiện trong listb. 
    Đảm bảo track không tồn tại ở cả 2 pool
    """
    remove_ids = {t.track_id for t in listb}
    return [t for t in lista if t.track_id not in remove_ids]
 
 
def remove_duplicate_stracks(stracksa, stracksb, iou_thresh=0.85):
    """
    Loại track trùng lặp giữa 2 pool dựa trên IoU.
    Giữ track nào có tracklet_len dài hơn (xuất hiện lâu hơn).
    """
    if not stracksa or not stracksb:
        return stracksa, stracksb
 
    # Tính IoU matrix
    boxesa = np.array([t.tlbr for t in stracksa], dtype=np.float32)
    boxesb = np.array([t.tlbr for t in stracksb], dtype=np.float32)
 
    iou = _iou_batch(boxesa, boxesb)
    pairs = np.where(iou > (iou_thresh))   # IoU > 0.85 → gần như trùng
 
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
    
# ------- Main class -------------------------------------------------
class DeepEIoU:
    def __init__(self,
                 track_high_thresh: float = 0.60,     # thresh cho các box vào first association (chặn dưới)
                 track_low_thresh: float = 0.1,       # thres cho low score box (chặn dưới)
                 new_track_thresh: float = 0.65,       # thres để tạo track mới
                 track_buffer: int = 60,              # vùng đệm, sau khi lost bằng này thì xóa track
                 match_thresh: float = 0.8,           # thresh cho first association
                 proximity_thresh: float = 0.7,       # thres về IOU trong stage 1
                 appearance_thresh: float = 0.25,     # appearance_thresh 
                 with_reid: bool = True):

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        BaseTrack.clear_count()
        
        self.frame_id = 0
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_time_lost = track_buffer

        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid

    def update(self, detections: List[Dict], frame_id: int) -> List[STrack]:
        """
        detections: list[{'tlbr': [x1,y1,x2,y2], 'score': float, 'feat': np.ndarray}]
        """
        self.frame_id = frame_id

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ====================== Phân loại detections ======================
        det_high = [d for d in detections if d['score'] >= self.track_high_thresh]
        det_low  = [d for d in detections if self.track_low_thresh <= d['score'] < self.track_high_thresh]

        # ====================== Tạo STrack từ detections ======================
        _detections: List[STrack] = []   #List các STrack, không còn là List các kết quả detection nữa
        # STrack(tlwh, score, feat)
        if det_high:
            for d in det_high:
                feat = d.get('feat') if self.with_reid else None
                x1, y1, x2, y2 = d['tlbr']
                tlwh = [x1, y1, x2-x1, y2-y1]
                _detections.append(STrack(tlwh, d['score'], feat))

        # ====================== Step 1: Phân biệt unconfirmed và tracked_stracks ======================
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]

        # ====================== Step 2: First association (High score + Iterative) ======================
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # Ghép lost strack và track_strack

        # siêu tham số của repo
        num_iteration = 2
        init_expand_scale = 0.7
        expand_scale_step = 0.1

        for iteration in range(num_iteration):
            cur_expand_scale = init_expand_scale + expand_scale_step * iteration    # mức mở rộng hiện tại

            # Expansion IoU distance
            ious_dists = eiou_distance(strack_pool, _detections, cur_expand_scale)
            ious_dists_mask = (ious_dists > self.proximity_thresh)      

            if self.with_reid:
                emb_dists = embedding_distance(strack_pool, _detections)
                emb_dists = emb_dists / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = _detections[idet]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id = False)
                    refind_stracks.append(track)

            # Cập nhật lại pool và detections chưa match
            strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            _detections = [_detections[i] for i in u_detection]

        # ====================== Step 3: Second association (Low score) ======================
        detections_second: List[STrack] = []
        if len(det_low) > 0:
            for d in det_low:
                feat = d.get('feat') if self.with_reid else None
                x1, y1, x2, y2 = d['tlbr']
                tlwh = [x1, y1, x2-x1, y2-y1]
                detections_second.append(STrack(tlwh, d['score'], feat))

        r_tracked_stracks = strack_pool
        dists = eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark lost cho những track không match
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ====================== Deal with unconfirmed tracks ======================
        if len(unconfirmed) > 0 and len(_detections) > 0:   # detections ở đây là remaining high sau iterative
            ious_dists = eiou_distance(unconfirmed, _detections, 0.5)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if self.with_reid:
                emb_dists = embedding_distance(unconfirmed, _detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

            for itracked, idet in matches:
                track = unconfirmed[itracked]
                det = _detections[idet]
                track.update(det, self.frame_id)
                activated_stracks.append(track)

            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        # ====================== Step 4: Init new stracks ======================
        for det in _detections:
            if det.score >= self.new_track_thresh:
                det.activate(frame_id)
                activated_stracks.append(det)
                
        # ====================== Step 5: Update state ======================
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ====================== Merge all ======================
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # Output
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks #List STrack
