# Tracklet.py
"""
Lớp đại diện cho một tracklet (đoạn track của một đối tượng) trong MOT.
Đây là cấu trúc dữ liệu chính mà GTALink sử dụng để thực hiện split và merge.
"""
class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None, feats=None):
        '''
        Khởi tạo một Tracklet.
        
        Parameters
        ----------
        track_id : int
            ID của tracklet.
        frames : list[int] hoặc int
            Danh sách các frame index (thường là 1-based) mà tracklet xuất hiện.
        scores : list[float] hoặc float
            Detection confidence score tương ứng với từng frame.
        bboxes : list[list] hoặc list
            Bounding box theo format [left, top, width, height] cho từng frame.
        feats : list[np.ndarray] 
            Danh sách feature vectors (đã L2-normalized) tương ứng với từng frame.
            Shape của mỗi feature tùy thuộc vào extractor
        '''
        self.track_id = track_id
        self.parent_id = track_id       # Dùng để theo dõi track gốc sau khi split/merge
        self.scores = scores if isinstance(scores, list) else [scores] if scores is not None else []
        self.times = frames if isinstance(frames, list) else [frames] if frames is not None else []
        self.bboxes = bboxes if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else [bboxes] if bboxes is not None else []
        self.features = feats if feats is not None else []

    def append_det(self, frame, score, bbox):
        '''
        Thêm một detection mới vào tracklet.
        bbox : list[float]
            [left, top, width, height]
        '''
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)

    def append_feat(self, feat):
        # Thêm feature vector cho detection mới nhất.
        self.features.append(feat)

    def extract(self, start, end):
        '''
        Trích xuất một sub-tracklet từ chỉ số start đến end (dùng trong splitting).
        
        Parameters
        ----------
        start : int
            Chỉ số bắt đầu (inclusive)
        end : int
            Chỉ số kết thúc (inclusive)
            
        Returns
        -------
        Tracklet
            Một Tracklet con chứa dữ liệu từ start đến end.
        '''
        subtrack = Tracklet(
            self.track_id,
            self.times[start:end + 1],
            self.scores[start:end + 1],
            self.bboxes[start:end + 1],
            self.features[start:end + 1] if self.features else None
        )
        return subtrack