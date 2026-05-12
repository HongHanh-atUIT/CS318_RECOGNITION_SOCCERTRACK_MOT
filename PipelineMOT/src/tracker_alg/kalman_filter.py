import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter(object):
    """
    Kalman Filter cho bounding box tracking.

    State space (8D): x, y, a, h, vx, vy, va, vh
        x, y : tọa độ trung tâm
        a    : aspect ratio (w/h)
        h    : height
        vx, vy, va, vh : vận tốc tương ứng

    NSA (Noise Scale Adaptive): scale measurement noise R theo detection score.
    Theo code gốc dyhBUPT: std = [(1 - confidence) * x for x in std]
    → scale từng phần tử std TRƯỚC khi tạo innovation_cov
    → đảm bảo projected_cov luôn positive definite
    """

    def __init__(self):
        ndim, dt = 4, 1.

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        Khởi tạo track từ measurement (x, y, a, h).
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean       = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        """
        Project state → measurement space.

        NSA: scale std theo (1 - confidence) TRƯỚC khi tạo innovation_cov.
        Cách này đảm bảo projected_cov = H@P@H.T + R luôn positive definite
        vì R = diag(std²) ≥ 0 với mọi std.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]

        # NSA: scale std theo (1 - confidence)
        # confidence cao → std nhỏ → R nhỏ → tin detection hơn
        std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov  = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return projected_mean, projected_cov + innovation_cov

    def multi_predict(self, mean, covariance):
        """Vectorized predict cho nhiều tracks."""
        N = len(mean)
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones(N),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones(N),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr        = np.square(np.r_[std_pos, std_vel]).T   # (N, 8)
        motion_cov = np.array([np.diag(s) for s in sqr])    # (N, 8, 8)
        mean       = (self._motion_mat @ mean.T).T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement, confidence=0.0):
        """
        Kalman update step với NSA noise scaling.
        confidence được truyền vào project() để scale R.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        innovation      = measurement - projected_mean
        new_mean        = mean + np.dot(innovation, kalman_gain.T)
        new_covariance  = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True,
            check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


# ─────────────────────────────────────────────────────────────
# KalmanFilterXY dùng cho trường hợp state là x,y pitch
# Dùng trong ByteTrack và StrongSort
# State space: [x, y, x', y']
# ─────────────────────────────────────────────────────────────
class KalmanFilterXY(KalmanFilter):
    """
    Biến thể stateless cho state [x, y, vx, vy] — 4D.
    Dùng khi use_project=True, measurement là tọa độ sân [x, y].
    Interface giữ nguyên: predict(mean, cov), update(mean, cov, measurement)
    """

    def __init__(self):
        # không gọi super().__init__() vì ta override toàn bộ
        self._motion_mat = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self._update_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """measurement: [x, y]"""
        mean = np.r_[measurement, np.zeros(2)]  # [x, y, 0, 0]
        std = [
            2 * self._std_weight_position,
            2 * self._std_weight_position,
            10 * self._std_weight_velocity,
            10 * self._std_weight_velocity,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position] * 2   # 2 phần tử là self._std_weight_position
        std_vel = [self._std_weight_velocity] * 2
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean       = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        std = [self._std_weight_position] * 2
        std = [(1 - confidence) * x for x in std]
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_mat @ mean
        projected_cov  = self._update_mat @ covariance @ self._update_mat.T
        return projected_mean, projected_cov + innovation_cov

    def multi_predict(self, mean, covariance):
        N = len(mean)
        std_pos = [self._std_weight_position * np.ones(N)] * 2
        std_vel = [self._std_weight_velocity * np.ones(N)] * 2
        sqr        = np.square(np.r_[std_pos, std_vel]).T   # (N, 4)
        motion_cov = np.array([np.diag(s) for s in sqr])    # (N, 4, 4)
        mean       = (self._motion_mat @ mean.T).T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance
    
    
# ─────────────────────────────────────────────────────────────
# KalmanFilterNew — dùng cho OC-SORT
# State space: [x, y, s, h, x', y', s']
# ─────────────────────────────────────────────────────────────

class KalmanFilterNew:
    """
    Kalman Filter dùng cho OC-SORT

    - Nếu new_kf = Fasle -> state có dạng (x, y, s, h, x', y', s') -> dimx = 7, dimz = 4
    - Nếu new_kf = True -> state có dạng (x, y) -> dimx = 4, dimz = 2
                            R và Q cũng được biến đổi để không có nhiễu theo kích thước
    Interface tương thích filterpy.kalman.KalmanFilter:
      self.x, self.P, self.F, self.H, self.R, self.Q
      .predict(Q=None), .update(z, R=None)
    """

    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))       # state vector
        self.P = np.eye(dim_x)              # covariance
        self.Q = np.eye(dim_x)              # process noise
        self.F = np.eye(dim_x)              # state transition
        self.H = np.zeros((dim_z, dim_x))  # observation model
        self.R = np.eye(dim_z)              # measurement noise

        self._I = np.eye(dim_x)

    def predict(self, F=None, Q=None):
        """Prediction step — dùng F, Q từ argument hoặc self."""
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R=None, H=None):
        """
        Update step.
        z = None → bỏ qua update (track bị lost)
        """
        if z is None:
            return

        if R is None:
            R = self.R
        if H is None:
            H = self.H

        z = np.atleast_2d(z)
        if z.shape == (1, self.dim_z):
            z = z.T

        y   = z - H @ self.x
        S   = H @ self.P @ H.T + R
        K   = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH   = self._I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def md_for_measurement(self, z) -> float:
        """Mahalanobis distance — dùng sau predict()."""
        z   = np.atleast_2d(z)
        if z.shape == (1, self.dim_z):
            z = z.T
        H   = self.H
        S   = H @ self.P @ H.T + self.R
        y   = z - H @ self.x
        md  = float(np.sqrt(y.T @ np.linalg.inv(S) @ y))
        return md