"""
basetrack.py
------------
Abstract class quản lý State, track_id, trạng thái của track.

Gộp từ 2 version:
  - Version bạn   : time_since_update là INSTANCE variable (đúng hơn)
  - Version leader : thêm mark_long_lost(), clear_count()

Fix:
  - time_since_update là instance variable → mỗi track có counter riêng
  - clear_count() reset _count về 0 để dùng lại giữa các pipeline
"""

import numpy as np
from collections import OrderedDict


class TrackState:
    New      = 0
    Tracked  = 1
    Lost     = 2
    LongLost = 3
    Removed  = 4


class BaseTrack:
    _count = 0

    track_id     = 0
    is_activated = False
    state        = TrackState.New

    history      = OrderedDict()
    features     = []
    curr_feature = None
    score        = 0
    start_frame  = 0
    frame_id     = 0

    # multi-camera (không dùng trong project này)
    location = (np.inf, np.inf)

    def __init__(self):
        # INSTANCE variable — mỗi track có counter riêng
        # tăng sau mỗi predict(), reset về 0 khi update/activate/re_activate
        self.time_since_update = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def clear_count():
        """Reset ID counter — gọi khi bắt đầu pipeline mới."""
        BaseTrack._count = 0

    # ── Abstract methods ────────────────────────────────────

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    # ── State transitions ───────────────────────────────────

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed