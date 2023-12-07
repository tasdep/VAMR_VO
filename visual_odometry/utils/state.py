#!/usr/bin/env python3


import numpy as np


class State:
    """
    Represents the state of a visual odometry system. This class encapsulates
    various numpy arrays to represent 3D landmarks, 2D keypoints, and camera poses,
    facilitating their management and update within a visual odometry pipeline.

    Attributes:
        _X (np.ndarray): A private 3xN array of 3D landmark locations.
        _P (np.ndarray): A private 2xN array of 2D keypoint locations corresponding to the landmarks.
        _C (np.ndarray): A private 2xM array of candidate 2D keypoint locations.
        _F (np.ndarray): A private 2xM array of the first observation of each candidate keypoint.
        _T (np.ndarray): A private 16xM array representing the pose of the camera for each candidate keypoint's first
                         observation. It stores a vectorized version of a 4x4 homogeneous transformation matrix.

    Methods provide functionality to update these attributes, ensuring data integrity and proper dimensionality.
    """

    def __init__(self) -> None:
        # 3xN array for 3D landmark locations
        self._X: np.ndarray = None

        # 2xN array for 2D keypoint locations corresponding to the landmarks
        self._P: np.ndarray = None

        # 2xM array for candidate 2D keypoint locations
        self._C: np.ndarray = None

        # 2xM array for the first observation of each candidate keypoint
        self._F: np.ndarray = None

        # 16xM array for the pose of the camera corresponding to the first
        # observation of each candidate keypoint. Represents a vecotirzed
        # version of a 4x4 homogenous transformation matrix of the format
        # [R | T]
        # [0 | 1]
        self._T: np.ndarray = None

    @property
    def X(self):
        return self._X

    @property
    def P(self):
        return self._P

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def T(self):
        return self._T

    def update_landmarks(self, new_X: np.ndarray, new_P: np.ndarray) -> None:
        """Update the 3D landmarks and their corresponding 2D keypoints."""
        if (
            new_P.shape[0] != 2
            or new_X.shape[0] != 3
            or new_P.shape[1] != new_X.shape[1]
        ):
            raise ValueError(
                f"Incorrect input shape. P must be 2xN (shape is {new_P.shape}) "
                f"and X must be 3xN (shape is {new_X.shape})."
            )
        self._X = new_X
        self._P = new_P

    def add_landmark(self, new_2D_point: np.ndarray, new_3D_point: np.ndarray) -> None:
        """Append a 2x1 2D keypoint to P and a 3x1 3D landmark to X."""
        if new_2D_point.shape != (2, 1) or new_3D_point.shape != (3, 1):
            raise ValueError(
                f"Incorrect shape for new points. 2D point must be 2x1 and 3D point must be 3x1."
                f"2D keypoint is shape {new_2D_point.shape} and 3D landmark is shape {new_3D_point.shape}"
            )
        self._P = np.hstack([self._P, new_2D_point])
        self._X = np.hstack([self._X, new_3D_point])

    def update_candidates(
        self, new_C: np.ndarray, new_F: np.ndarray, new_T: np.ndarray
    ) -> None:
        """Update the candidate keypoints, their first observations, and corresponding camera poses."""
        if (
            new_C.shape[0] != 2
            or new_F.shape[0] != 2
            or new_T.shape[0] != 16
            or new_C.shape[1] != new_F.shape[1]
            or new_C.shape[1] != new_T.shape[1]
        ):
            raise ValueError(
                f"Incorrect input shape. C must be 2xM (shape is {new_C.shape}), "
                f"F must be 2xM (shape is {new_F.shape}) and T much be 16xM (shape is {new_T.shape})."
            )
        self._C = new_C
        self._F = new_F
        self._T = new_T
