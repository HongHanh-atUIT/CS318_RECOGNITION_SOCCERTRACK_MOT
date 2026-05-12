"""
tracker.py
Wrapper thống nhất cho các thuật toán MOT: ByteTrack, StrongSort, OC-Sort, DeepEIoU.
 
Input format chuẩn từ detector + extractor:
    [{'tlbr': [x1,y1,x2,y2], 'score': float, 'feat': np.ndarray}, ...]
 
Output: List[STrack] — mỗi track có .track_id và .tlbr (hoặc .last_tlbr)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional


class Tracker:
    """
    Wrapper dùng chung cho tất cả thuật toán MOT.
    
    Usage:
        tracker = Tracker(algorithm='deepeiou', **kwargs)
        active_tracks = tracker.update(detections, frame_id)
    
    Supported algorithms: 'deepeiou', 'bytetrack', 'strongsort', 'ocsort'
    """
    SUPPORTED = ('deepeiou', 'bytetrack', 'strongsort', 'ocsort')
    def __init__(self, use_project = False, algorithm: str = 'deepeiou', **kwargs):
        """
        algorithm : tên thuật toán (case-insensitive)
        kwargs    : tham số truyền thẳng vào constructor của tracker tương ứng
        """
        algo = algorithm.lower().replace('-', '').replace('_', '')
        kwargs['use_project'] = use_project
        if algo == 'deepeiou':
            if use_project:
                raise ValueError("Deep EIOU không thể sử dụng phép chiếu! Sử dụng Deep EIOU mặc định")
            from tracker_alg.deepeiou import DeepEIoU
            self._tracker = DeepEIoU(**kwargs)
        elif algo == 'bytetrack':
            from tracker_alg.bytetrack import ByteTrack
            self._tracker = ByteTrack(**kwargs)
        elif algo == 'strongsort':
            from tracker_alg.strongsort import StrongSort
            self._tracker = StrongSort(**kwargs)
        elif algo == 'ocsort':
            from tracker_alg.ocsort import OCSort
            self._tracker = OCSort(**kwargs)
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {self.SUPPORTED}"
            )
 
        self.algorithm = algo
        self.use_project = use_project
 
    def update(self, detections: List[Dict], frame_id: int):
        """
        Chạy một bước tracking cho frame hiện tại. Nếu có sử dụng phép chiếu, thay 'tlbr' bằng 'xy'
        'xy' là tọa độ của cầu thủ trên sân bóng
        
        Parameters
        ----------
        detections : list of dict
            Mỗi dict có:
                'tlbr'  : [x1, y1, x2, y2]  (float)
                'score' : float
                'feat'  : np.ndarray hoặc None
        frame_id : int  (bắt đầu từ 1)
 
        Returns
        -------
        List[STrack]  — các track đang active tại frame này
        """
        return self._tracker.update(detections, frame_id)