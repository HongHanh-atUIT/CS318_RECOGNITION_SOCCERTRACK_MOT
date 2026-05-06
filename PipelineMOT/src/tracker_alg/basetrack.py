"""
Abstract Class, chỉ quản lý State, track_id, trạng thái, history của track

FIX v2:
- time_since_update là class variable → đổi thành instance variable,
  tự tăng khi predict(), reset về 0 khi update()/activate()/re_activate()
- end_frame property giữ nguyên (return self.frame_id là đúng)
- clear_count() reset _count về 0 để dùng lại giữa các pipeline
"""
import numpy as np
from collections import OrderedDict


class TrackState(object):
    New      = 0
    Tracked  = 1
    Lost     = 2
    LongLost = 3
    Removed  = 4


class BaseTrack(object):
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

    # multi-camera
    location = (np.inf, np.inf)

    def __init__(self):
        # FIX: time_since_update là INSTANCE variable, không phải class variable
        # Mỗi track có counter riêng, tăng sau mỗi predict(), reset khi được update
        self.time_since_update = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0