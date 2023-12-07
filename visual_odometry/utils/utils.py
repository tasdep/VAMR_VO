import numpy as np


def create_homogeneous_matrix(R: np.ndarray, t: np.ndarray):
    """
    Concatenate rotation matrix R and translation vector t into a 4x4 homogeneous transformation matrix.

    Args:
    - R (np.ndarray): 3x3 rotation matrix.
    - t (np.ndarray): 3x1 translation vector.

    Returns:
    - np.ndarray: 4x4 homogeneous transformation matrix.
    """
    homogeneous_matrix: np.ndarray = np.eye(4)
    homogeneous_matrix[:3, :3] = R
    homogeneous_matrix[:3, 3] = t.flatten()
    return homogeneous_matrix
