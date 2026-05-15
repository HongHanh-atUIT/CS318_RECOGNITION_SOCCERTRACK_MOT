"""
kalman_filter.py
----------------
Ba class Kalman Filter dùng trong toàn bộ pipeline:

  KalmanFilter      — state [cx,cy,a,h, vx,vy,va,vh]  8D  — ByteTrack/StrongSORT gốc
  KalmanFilterXY    — state [x, y, dx, dy]             4D  — use_project=True
  KalmanFilterNew   — state [x,y,s,r,vx,vy,vs] / [x,y,vx,vy]  — OC-SORT
"""

import numpy as np
import scipy.linalg


chi2inv95 = {1:3.8415,2:5.9915,3:7.8147,4:9.4877,5:11.070,
             6:12.592,7:14.067,8:15.507,9:16.919}


# ─────────────────────────────────────────────────────────────
# KalmanFilter — state [cx,cy,a,h,vx,vy,va,vh]  (gốc ByteTrack/StrongSORT)
# ─────────────────────────────────────────────────────────────

class KalmanFilter:
    """State: [cx, cy, a, h, vx, vy, va, vh]  (8D)."""

    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean     = np.r_[mean_pos, mean_vel]
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
        std_pos = [self._std_weight_position * mean[3]] * 2 + [1e-2, self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3]] * 2 + [1e-5, self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean       = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        std = [self._std_weight_position * mean[3]] * 2 + [1e-1, self._std_weight_position * mean[3]]
        std = [(1 - confidence) * x for x in std]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov  = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return projected_mean, projected_cov + innovation_cov

    def multi_predict(self, mean, covariance):
        N = len(mean)
        std_pos = [self._std_weight_position * mean[:, 3]] * 2 + \
                  [1e-2 * np.ones(N), self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 3]] * 2 + \
                  [1e-5 * np.ones(N), self._std_weight_velocity * mean[:, 3]]
        sqr        = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.array([np.diag(s) for s in sqr])
        mean       = (self._motion_mat @ mean.T).T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement, confidence=0.0):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation     = measurement - projected_mean
        new_mean       = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


# ─────────────────────────────────────────────────────────────
# KalmanFilterXY — state [x, y, dx, dy]  4D  (use_project=True)
# ─────────────────────────────────────────────────────────────

class KalmanFilterXY:
    """
    Kalman Filter cho pitch-space tracking.

    State  (4D): [x, y, dx, dy]
    Measurement (2D): [x, y]

    NSA: scale measurement noise R theo detection confidence.
    """

    def __init__(self, std_weight_position=0.05, std_weight_velocity=0.00625):
        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity

        # F: state transition (4×4)
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # H: observation (2×4)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

    def initiate(self, measurement):
        """measurement: [x, y]"""
        mean = np.array([measurement[0], measurement[1], 0.0, 0.0])
        std  = [2*self._std_pos, 2*self._std_pos,
                10*self._std_vel, 10*self._std_vel]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        Q = np.diag(np.square([self._std_pos, self._std_pos,
                                self._std_vel, self._std_vel]))
        mean_pred = self._F @ mean
        cov_pred  = self._F @ covariance @ self._F.T + Q
        return mean_pred, cov_pred

    def project(self, mean, covariance, confidence=0.0):
        std = [(1 - confidence) * self._std_pos] * 2
        R   = np.diag(np.square(std))
        proj_mean = self._H @ mean
        proj_cov  = self._H @ covariance @ self._H.T + R
        return proj_mean, proj_cov

    def multi_predict(self, means, covariances):
        N   = len(means)
        Q   = np.diag(np.square([self._std_pos, self._std_pos,
                                   self._std_vel, self._std_vel]))
        means_pred = (self._F @ means.T).T
        covs_pred  = self._F @ covariances @ self._F.T + Q[np.newaxis]
        return means_pred, covs_pred

    def update(self, mean, covariance, measurement, confidence=0.0):
        proj_mean, proj_cov = self.project(mean, covariance, confidence)
        chol, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve(
            (chol, lower), (covariance @ self._H.T).T,
            check_finite=False).T
        innovation     = measurement - proj_mean
        new_mean       = mean + K @ innovation
        new_covariance = covariance - K @ proj_cov @ K.T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements):
        """Mahalanobis distance, 2 DOF."""
        proj_mean, proj_cov = self.project(mean, covariance)
        chol = np.linalg.cholesky(proj_cov)
        d    = measurements - proj_mean
        z    = scipy.linalg.solve_triangular(
            chol, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


# ─────────────────────────────────────────────────────────────
# KalmanFilterNew — dùng cho OC-SORT
# ─────────────────────────────────────────────────────────────

class KalmanFilterNew:
    """
    Kalman Filter dùng cho OC-SORT.
    Interface tương thích filterpy: self.x, self.P, self.F, self.H, self.R, self.Q
    .predict(Q=None), .update(z, R=None)
    """

    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self._I = np.eye(dim_x)

    def predict(self, F=None, Q=None):
        if F is None: F = self.F
        if Q is None: Q = self.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R=None, H=None):
        if z is None:
            return
        if R is None: R = self.R
        if H is None: H = self.H
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
        z = np.atleast_2d(z)
        if z.shape == (1, self.dim_z):
            z = z.T
        S  = self.H @ self.P @ self.H.T + self.R
        y  = z - self.H @ self.x
        return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))
