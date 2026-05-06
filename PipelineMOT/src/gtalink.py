"""
GTALink.py
----------
Offline refinement module cho MOT pipeline.

Gồm hai bước chính:
  1. Split  — phát hiện ID-switch bên trong một tracklet và tách ra
  2. Connect (Merge) — ghép các tracklet thuộc cùng một đối tượng lại

Cách dùng:
    refiner = GTALink(use_split=True, use_connect=True, ...)
    refined = refiner.refine(all_tracks)   # all_tracks từ run_mot()
"""

import numpy as np
import torch

from collections import defaultdict
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from .tracklet import Tracklet


# ─────────────────────────────────────────────
# Các hàm tiện ích nội bộ
# ─────────────────────────────────────────────

def _find_consecutive_segments(track_times):
    """
    Tìm các đoạn liên tiếp trong danh sách frame times.

    Ví dụ: [1,2,3, 7,8, 12] → [(0,2), (3,4), (5,5)]

    Returns
    -------
    list of (start_idx, end_idx)
    """
    segments = []
    start = end = 0
    for i in range(1, len(track_times)):
        if track_times[i] == track_times[end] + 1:
            end = i
        else:
            segments.append((start, end))
            start = end = i
    segments.append((start, end))
    return segments


def _query_subtracks(seg1, seg2, track1, track2):
    """
    Ghép các segment của hai track thành chuỗi sub-tracklet sắp xếp theo thời gian.

    Logic: duyệt song song seg1 và seg2; tại mỗi bước so sánh frame bắt đầu
    để xác định thứ tự (track nào đến trước thì append trước).
    Phần dư (nếu một list hết trước) được append thêm nếu đủ dài (>30 frame).

    Returns
    -------
    list of Tracklet
    """
    subtracks = []

    while seg1 and seg2:
        s1_start, s1_end = seg1[0]
        s2_start, s2_end = seg2[0]

        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        s1_startFrame = track1.times[s1_start]
        s2_startFrame = track2.times[s2_start]

        if s1_startFrame < s2_startFrame:
            # track1 đến trước → track1 phải kết thúc trước khi track2 bắt đầu
            assert track1.times[s1_end] <= s2_startFrame
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            # track2 đến trước
            assert s1_startFrame >= track2.times[s2_end]
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)

        seg1.pop(0)
        seg2.pop(0)

    # Xử lý phần dư (nếu một track còn segment thừa)
    seg_remain   = seg1 if seg1 else seg2
    track_remain = track1 if seg1 else track2
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if (s_end - s_start) < 30:          # bỏ đoạn quá ngắn
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)

    return subtracks


def _get_spatial_constraints(tracklets, factor):
    """
    Tính khoảng cách không gian tối đa (x, y) cho phép khi merge.

    Duyệt toàn bộ bounding box, lấy range của tọa độ tâm rồi nhân với factor.

    Returns
    -------
    (max_x_range, max_y_range) : float
    """
    min_x = min_y =  float('inf')
    max_x = max_y = -float('inf')

    for track in tracklets.values():
        for bbox in track.bboxes:
            x, y, w, h = bbox[:4]
            cx = x + w / 2
            cy = y + h / 2
            min_x, max_x = min(min_x, cx), max(max_x, cx)
            min_y, max_y = min(min_y, cy), max(max_y, cy)

    return abs(max_x - min_x) * factor, abs(max_y - min_y) * factor


# ─────────────────────────────────────────────
# Tính khoảng cách giữa các tracklet
# ─────────────────────────────────────────────

def _get_distance(track1, track2):
    """
    Tính cosine distance trung bình giữa hai tracklet dựa trên feature vectors.

    - Nếu hai track overlap về thời gian → distance = 1 (tối đa, không merge).
    - Ngược lại → tính cosine distance trung bình mọi cặp feature.

    Returns
    -------
    float trong [0, 1]
    """
    # Hai track cùng ID hoặc overlap thời gian → không thể merge
    if track1.track_id != track2.track_id:
        if set(track1.times) & set(track2.times):
            return 1.0
    # calculate cosine distance between two tracks based on features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1 = torch.tensor(np.stack(track1.features), dtype=torch.float32).to(device)
    f2 = torch.tensor(np.stack(track2.features), dtype=torch.float32).to(device)

    # Cosine similarity ma trận: (N1, N2)
    cos_sim_Numerator = torch.matmul(f1, f2.T)
    f1_dist = torch.norm(f1, dim=1, keepdim=True)
    f2_dist = torch.norm(f2, dim=1, keepdim=True)
    cos_sim_Denominator = torch.matmul(f1_dist, f2_dist.T)
    cos_Dist = 1 - cos_sim_Numerator / cos_sim_Denominator
    
    total_cos_Dist = cos_Dist.sum()
    result = total_cos_Dist / (len(f1) * len(f2))
    
    return result


def _get_distance_matrix(tracklets):
    """
    Xây dựng ma trận khoảng cách NxN cho toàn bộ tracklet trong một sequence.

    Tận dụng tính đối xứng: Dist[i][j] = Dist[j][i].
    """
    tracks = list(tracklets.values())
    n = len(tracks)
    Dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = _get_distance(tracks[i], tracks[j])
            Dist[i][j] = Dist[j][i] = d
    return Dist


# ─────────────────────────────────────────────
# Kiểm tra ràng buộc không gian khi merge
# ─────────────────────────────────────────────

def _check_spatial_constraints(trk_1, trk_2, max_x_range, max_y_range):
    """
    Kiểm tra xem hai tracklet có thỏa ràng buộc không gian để merge không.

    Ý tưởng: lấy vị trí cuối của sub-tracklet trước và vị trí đầu của sub-tracklet
    sau (theo thời gian), kiểm tra khoảng cách tâm bbox có trong ngưỡng cho phép.

    Returns
    -------
    bool : True nếu đủ điều kiện không gian để merge
    """
    
    seg1 = _find_consecutive_segments(trk_1.times)
    seg2 = _find_consecutive_segments(trk_2.times)
    subtracks = _query_subtracks(seg1, seg2, trk_1, trk_2)

    subtrack_prev = subtracks.pop(0)
    while subtracks:
        subtrack_curr = subtracks.pop(0)

        # Bỏ qua nếu hai sub-track liên tiếp thuộc cùng track gốc
        if subtrack_prev.parent_id == subtrack_curr.parent_id:
            subtrack_prev = subtrack_curr
            continue

        # Tọa độ tâm: cuối track trước và đầu track sau
        x1, y1, w1, h1 = subtrack_prev.bboxes[-1][:4]
        x2, y2, w2, h2 = subtrack_curr.bboxes[0][:4]
        dx = abs((x1 + w1 / 2) - (x2 + w2 / 2))
        dy = abs((y1 + h1 / 2) - (y2 + h2 / 2))

        if dx > max_x_range or dy > max_y_range:
            return False    # vượt ngưỡng → không merge

        subtrack_prev = subtrack_curr

    return True


# ─────────────────────────────────────────────
# Bước SPLIT
# ─────────────────────────────────────────────

def _detect_id_switch(embs, eps, min_samples, max_clusters):
    """
    Dùng DBSCAN để phát hiện ID-switch bên trong một tracklet.

    Nếu tìm được > 1 cluster thực → có thể có ID-switch.
    Noise points được gán vào cluster gần nhất.
    Nếu số cluster > max_clusters → merge cluster gần nhất cho đến khi đủ.

    Returns
    -------
    (id_switch_detected : bool, labels : np.ndarray)
    """
    # Giảm mẫu nếu quá lớn (> 15000 embedding)
    if len(embs) > 15000:
        embs = embs[1::2]

    embs = np.stack(embs)
    embs_scaled = StandardScaler().fit_transform(embs)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs_scaled)
    labels = db.labels_.copy()

    # Count the number of clusters (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    # Gán noise vào cluster gần nhất (nếu có nhiều hơn 1 cluster)
    if -1 in labels and len(unique_labels) > 1:
        cluster_centers = np.array([embs_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels])
        
        # Assign noise points to the nearest cluster
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric='cosine')
            nearest_cluster = np.argmin(distances)
            labels[idx] = list(unique_labels)[nearest_cluster]
            
    n_clusters = len(unique_labels)

    # Merge cluster nếu vượt quá max_clusters
    if max_clusters and n_clusters > max_clusters:
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == label].mean(axis=0) for label in unique_labels])
            distance_matrix = cdist(cluster_centers, cluster_centers, metric='cosine')
            np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
            
            # Find the closest pair of clusters
            min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            cluster_to_merge_1, cluster_to_merge_2 = unique_labels[min_dist_idx[0]], unique_labels[min_dist_idx[1]]

            # Merge the clusters
            labels[labels == cluster_to_merge_2] = cluster_to_merge_1
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def _split_tracklets(tracklets, eps, min_samples, max_k, len_thres):
    """
    Bước Split: với mỗi tracklet đủ dài, chạy DBSCAN để kiểm tra ID-switch.
    Nếu phát hiện → tách thành nhiều tracklet con.

    Parameters
    ----------
    tracklets  : dict {track_id: Tracklet}
    eps        : float  — tham số DBSCAN
    min_samples: int    — tham số DBSCAN
    max_k      : int    — số cluster tối đa
    len_thres  : int    — tracklet ngắn hơn ngưỡng này sẽ bỏ qua split

    Returns
    -------
    dict {track_id: Tracklet}  — tracklet sau khi split (len >= len(tracklets))
    """
    new_id = max(tracklets.keys()) + 1
    result = defaultdict()

    for tid in tqdm(sorted(tracklets.keys()), desc="Splitting tracklets"):
        trklet = tracklets[tid]

        # Tracklet quá ngắn → giữ nguyên
        if len(trklet.times) < len_thres:
            result[tid] = trklet
            continue

        embs   = np.stack(trklet.features)
        frames = np.array(trklet.times)
        bboxes = np.stack(trklet.bboxes)
        scores = np.array(trklet.scores)

        id_switch, clusters = _detect_id_switch(embs, eps, min_samples, max_k)

        if not id_switch:
            result[tid] = trklet          # không phát hiện → giữ nguyên
        else:
            # Tách từng cluster thành tracklet riêng
            unique_labels = set(clusters)
            for lbl in unique_labels:
                if lbl == -1:
                    continue              # bỏ noise
                mask = clusters == lbl
                result[new_id] = Tracklet(
                    new_id,
                    frames[mask].tolist(),
                    scores[mask].tolist(),
                    bboxes[mask].tolist(),
                    feats=embs[mask].tolist()
                )
                new_id += 1

    assert len(result) >= len(tracklets)
    return result


# ─────────────────────────────────────────────
# Bước CONNECT (Merge)
# ─────────────────────────────────────────────

def _merge_tracklets(tracklets, Dist, max_x_range, max_y_range, merge_dist_thres):
    """
    Bước Connect: dùng Hierarchical Clustering để ghép các tracklet gần nhau.

    Thuật toán:
      Lặp cho đến khi không còn cặp nào có distance < merge_dist_thres:
        1. Tìm cặp (i, j) có distance nhỏ nhất.
        2. Kiểm tra ràng buộc không gian.
           - Thỏa → merge track j vào track i, cập nhật Dist.
           - Không thỏa → đặt Dist[i,j] = threshold (không xét lại).

    Parameters
    ----------
    tracklets       : dict {track_id: Tracklet}
    Dist            : np.ndarray NxN — ma trận khoảng cách
    max_x_range     : float
    max_y_range     : float
    merge_dist_thres: float

    Returns
    -------
    dict {track_id: Tracklet}
    """
    idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

    diag_mask     = np.eye(Dist.shape[0], dtype=bool)
    off_diag_mask = ~diag_mask # ngoài đường chéo chính

    while np.any(Dist[off_diag_mask] < merge_dist_thres):
        # Tìm cặp có distance nhỏ nhất (ngoài đường chéo)
        flat_idx = np.argmin(Dist[off_diag_mask])
        rows, cols = np.where(off_diag_mask)
        i, j = rows[flat_idx], cols[flat_idx]   #track1_idx, track2_idx

        assert Dist[i, j] == Dist[j, i]

        track_i = tracklets[idx2tid[i]] # lấy track
        track_j = tracklets[idx2tid[j]]

        if _check_spatial_constraints(track_i, track_j, max_x_range, max_y_range):  # thỏa ràng buộc không gian
            # ── Merge track_j vào track_i ──
            track_i.features += track_j.features
            track_i.times    += track_j.times
            track_i.bboxes   += track_j.bboxes

            tracklets[idx2tid[i]] = track_i
            tracklets.pop(idx2tid[j])

            # Xóa hàng/cột của track_j khỏi Dist
            Dist = np.delete(Dist, j, axis=0)   #row
            Dist = np.delete(Dist, j, axis=1)

            # Cập nhật ánh xạ idx → track_id
            idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}

            # Cập nhật hàng/cột của track_i trong Dist
            for k in range(Dist.shape[0]):
                d = _get_distance(tracklets[idx2tid[i]], tracklets[idx2tid[k]])
                Dist[i, k] = Dist[k, i] = d

            # Cập nhật lại mask sau khi Dist thay đổi kích thước
            diag_mask     = np.eye(Dist.shape[0], dtype=bool)
            off_diag_mask = ~diag_mask
        else:
            # Không thỏa không gian → đặt bằng threshold để không xét lại
            Dist[i, j] = Dist[j, i] = merge_dist_thres

    return tracklets


# ─────────────────────────────────────────────
# Chuyển đổi định dạng all_tracks ↔ Tracklet
# ─────────────────────────────────────────────
def _all_tracks_to_tracklets(all_tracks):
    """
    Chuyển `all_tracks` (output của run_mot) sang dict {track_id: Tracklet}.
 
    all_tracks phải có trường 'feats' được run_mot điền sẵn (1-1 với 'frames').
 
    Parameters
    ----------
    all_tracks : dict
        {
            track_id: {
                'boxes' : list[box | None]  — box [x1,y1,x2,y2] hoặc None theo frame_idx,
                'frames': list[int]         — frame_idx thực có box (không None),
                'feats' : list[np.ndarray]  — embedding tương ứng 1-1 với 'frames'
            }
        }
 
    Returns
    -------
    dict {track_id: Tracklet}
    """
    tracklets = {}
    for tid, data in all_tracks.items():
        frames = data['frames']         # các frame_idx thực có box
        feats  = data['feats']          # embeddings tương ứng 1-1 với frames
        assert len(frames) == len(feats), \
            f"Track {tid}: số frame ({len(frames)}) != số feat ({len(feats)})"
 
        # Lấy box tương ứng với từng frame thực (bỏ qua None)
        # 'boxes' được index theo frame_idx → dùng frames để tra
        frame2box = {f: data['boxes'][f] for f in frames}
 
        # Chuyển [x1,y1,x2,y2] → [x,y,w,h] (format Tracklet dùng)
        bboxes_xywh = []
        for f in frames:
            x1, y1, x2, y2 = frame2box[f]
            bboxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
 
        scores = [1.0] * len(frames)   # score không dùng trong GTALink → để 1.0
 
        tracklets[tid] = Tracklet(tid, frames, scores, bboxes_xywh, feats=feats)
    return tracklets
 

def _tracklets_to_all_tracks(tracklets, max_frame):
    """
    Chuyển dict {track_id: Tracklet} về định dạng `all_tracks` chuẩn của run_mot.
 
    - 'boxes' được index theo frame_idx (giống all_tracks gốc): None nếu track không có box frame đó.
    - 'frames' chỉ chứa frame_idx thực có box.
    - Không gán lại track_id; giữ nguyên ID sau refinement.
 
    Parameters
    ----------
    tracklets : dict {track_id: Tracklet}
    max_frame : int — frame_idx lớn nhất của video gốc, truyền từ refine() để
                      đảm bảo list boxes có đúng độ dài, không phụ thuộc vào
                      frame nào còn sót lại sau split/merge.
 
    Returns
    -------
    dict {track_id: {'boxes': list[box | None], 'frames': list[int]}}
    """
    if not tracklets:
        return {}
 
    all_tracks = {}
    for tid, track in tracklets.items():
        # Tạo dict tra cứu nhanh frame → bbox (tránh .index() O(n) trong vòng lặp)
        frame2bbox = {}
        for k, f in enumerate(track.times):
            x, y, w, h = track.bboxes[k]
            frame2bbox[f] = [x, y, x + w, y + h]   # trả về x1y1x2y2
 
        boxes      = []
        frames_out = []
        for f in range(max_frame + 1):
            if f in frame2bbox:
                boxes.append(frame2bbox[f])
                frames_out.append(f)
            else:
                boxes.append(None)
 
        all_tracks[tid] = {'boxes': boxes, 'frames': frames_out}
    return all_tracks
 
 
# ─────────────────────────────────────────────
# Class GTALink (interface chính)
# ─────────────────────────────────────────────
 
class GTALink:
    """
    Offline tracklet refinement sử dụng thuật toán GTALink (Split + Connect).
 
    Nhận `all_tracks` từ run_mot (đã có trường 'feats' được điền sẵn).
 
    Parameters
    ----------
    use_split       : bool  — có chạy bước Split không
    use_connect     : bool  — có chạy bước Connect/Merge không
    eps             : float — epsilon cho DBSCAN (split)
    min_samples     : int   — min_samples cho DBSCAN (split)
    max_k           : int   — số cluster tối đa sau split
    min_len         : int   — tracklet ngắn hơn ngưỡng này sẽ bỏ qua split
    spatial_factor  : float — nhân với range tọa độ để lấy ngưỡng không gian
    merge_dist_thres: float — ngưỡng cosine distance để merge
    """
 
    def __init__(
        self,
        use_split        = True,
        use_connect      = True,
        eps              = 0.7,
        min_samples      = 10,
        max_k            = 3,
        min_len          = 30,
        spatial_factor   = 1.0,
        merge_dist_thres = 0.4,
    ):
        if not use_split and not use_connect:
            raise ValueError("Phải bật ít nhất một trong hai: use_split hoặc use_connect.")
 
        self.use_split        = use_split
        self.use_connect      = use_connect
        self.eps              = eps
        self.min_samples      = min_samples
        self.max_k            = max_k
        self.min_len          = min_len
        self.spatial_factor   = spatial_factor
        self.merge_dist_thres = merge_dist_thres
 
    def refine(self, all_tracks):
        """
        Chạy GTALink refinement trên all_tracks trả về bởi run_mot().
 
        all_tracks phải có trường 'feats' (run_mot tự điền khi extractor_refine được truyền vào).
 
        Parameters
        ----------
        all_tracks : dict
            {
                track_id: {
                    'boxes' : list[box | None],
                    'frames': list[int],
                    'feats' : list[np.ndarray]
                }
            }
 
        Returns
        -------
        dict — cùng định dạng all_tracks (không có 'feats') sau khi đã refinement
        """
        # Chuyển sang Tracklet (feats đọc thẳng từ all_tracks['feats'])
        max_frame = max(f for data in all_tracks.values() for f in data['frames'])
        tracklets = _all_tracks_to_tracklets(all_tracks)
 
        # Tính ràng buộc không gian dựa trên toàn bộ tracklet
        max_x_range, max_y_range = _get_spatial_constraints(tracklets, self.spatial_factor)
 
        # ── Bước 1: Split ──
        if self.use_split:
            print(f"[GTALink] Trước split: {len(tracklets)} tracklets")
            tracklets = _split_tracklets(
                tracklets,
                eps         = self.eps,
                min_samples = self.min_samples,
                max_k       = self.max_k,
                len_thres   = self.min_len,
            )
            print(f"[GTALink] Sau  split: {len(tracklets)} tracklets")
 
        # ── Bước 2: Connect (Merge) ──
        if self.use_connect:
            Dist = _get_distance_matrix(tracklets)
            print(f"[GTALink] Trước merge: {len(tracklets)} tracklets")
            tracklets = _merge_tracklets(
                tracklets,
                Dist,
                max_x_range      = max_x_range,
                max_y_range      = max_y_range,
                merge_dist_thres = self.merge_dist_thres,
            )
            print(f"[GTALink] Sau  merge: {len(tracklets)} tracklets")
 
        # Chuyển về định dạng all_tracks (không kèm feats)
        return _tracklets_to_all_tracks(tracklets, max_frame)