"""
tracker.py — Wrapper chỉ cho DeepSort.

Pipeline: YOLOv5 → OSNet → DeepSort

use_project=True  : Kalman state [x,y,dx,dy] trên pitch, Euclidean distance
use_project=False : Kalman state gốc (pixel), IoU + cosine distance  ← khuyến nghị
"""

import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


class Tracker:

    def __init__(
        self,
        algorithm:   str  = "deepsort",
        use_project: bool = False,
        H:     Optional[np.ndarray] = None,
        H_inv: Optional[np.ndarray] = None,
        frame_w: int = 1920,
        frame_h: int = 1080,
        **kwargs,
    ):
        if algorithm.lower().replace("-", "").replace("_", "") != "deepsort":
            raise ValueError(f"Chỉ hỗ trợ 'deepsort'. Nhận được: '{algorithm}'")

        if use_project and H is not None and H_inv is None:
            H_inv = np.linalg.inv(H)

        from tracker_alg.deepsort import DeepSort
        self._tracker = DeepSort(
            use_project=use_project,
            H=H, H_inv=H_inv,
            frame_w=frame_w, frame_h=frame_h,
            **kwargs,
        )

        self.algorithm   = "deepsort"
        self.use_project = use_project
        print(f"[Tracker] deepsort initialized | use_project={use_project}")

    def update(self, detections, frame_id: int):
        return self._tracker.update(detections, frame_id)