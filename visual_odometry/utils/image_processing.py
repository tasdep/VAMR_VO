import numpy as np
import cv2
import matplotlib.pyplot as plt
import params.params as params


def run_harris_detector(img: np.ndarray, visualise: bool = False, print_stats: bool = False):
    """
    Detects keypoints using the Harris corner detector on the given image.

    Parameters:
    - img: The input grayscale image.
    - visualise: Flag to visualize the detected keypoints.
    - print_stats: Flag to print statistics about the detected keypoints.

    Returns:
    - A Nx2 array containing N keypoint coordinates (row, column) where corners are detected.
    """
    harris_params = {
        "blockSize": params.HARRIS_BLOCK_SIZE,
        "ksize": params.HARRIS_SOBEL_SIZE,
        "k": params.HARRIS_K,
    }
    corners: np.ndarray = cv2.cornerHarris(img, **harris_params)

    # extract keypoints from corner detector
    keypoints: np.ndarray = np.argwhere(corners > params.KEYPOINT_THRESHOLD * corners.max())

    if print_stats:
        print(f"{keypoints.shape=}")

    if visualise:
        fig, axs = plt.subplots(1, 1)
        axs.imshow(img, cmap="gray")
        axs.plot(keypoints[:, 1], keypoints[:, 0], "rx")
        plt.show()

    return keypoints


def patch_describe_keypoints(img: np.ndarray, keypoints: np.ndarray, r: int) -> np.ndarray:
    """
    Extracts image patch descriptors centered around keypoints.

    Parameters:
    - img: The input image.
    - keypoints: A Nx2 matrix containing N keypoint coordinates (row, column).
    - r: The patch "radius." The size of each square patch is (2r+1)x(2r+1).

    Returns:
    - A Nx(2r+1)^2 matrix of image patch vectors. Each row corresponds to a keypoint,
      and the patch vector is a flattened (2r+1)x(2r+1) square region centered around the keypoint.
      If keypoints is None, an empty matrix with shape (0, (2r+1)^2) is returned.
    """
    if keypoints is None:
        return np.ndarray((0, (2 * r + 1) ** 2))
    N: int = keypoints.shape[0]
    descriptors: np.ndarray = np.zeros([N, (2 * r + 1) ** 2])
    padded: np.ndarray = np.pad(img, [(r, r), (r, r)], mode="constant", constant_values=0)

    for i in range(N):
        kp: np.ndarray = keypoints[i, :].astype(int) + r
        descriptors[i, :] = padded[(kp[0] - r) : (kp[0] + r + 1), (kp[1] - r) : (kp[1] + r + 1)].flatten()

    return descriptors


def triangulate_points_wrapper(T1: np.ndarray, T2: np.ndarray, K: np.ndarray, points_1: np.ndarray, points_2: np.ndarray):
    """
    Triangulates 3D points from corresponding 2D points in two camera views.

    Parameters:
    - T1: The transformation matrix (4x4) of the first camera.
    - T2: The transformation matrix (4x4) of the second camera.
    - K: The camera intrinsic matrix.
    - points_1: 2xN array of 2D pixel coordinates in the first image.
    - points_2: 2xN array of corresponding 2D pixel coordinates in the second image.

    Returns:
    - A 3xN array containing the triangulated 3D coordinates of the points.
    """
    # dehomogenize transforms
    T1 = T1[0:3, :]
    T2 = T2[0:3, :]
    M1: np.ndarray = K @ T1
    M2: np.ndarray = K @ T2
    points_3D: np.ndarray = cv2.triangulatePoints(M1, M2, points_1, points_2)
    # dehomgenize
    points_3D: np.ndarray = points_3D[0:3, :] / points_3D[3, :]
    return points_3D


# IDEA FOR NON LINEAR REPROJECTION OPTIMISATION
import numpy as np
from scipy.optimize import least_squares

# Assuming you have the following variables from your previous code
# K1, K2: Intrinsic matrices for cameras 1 and 2
# dist_coeffs1, dist_coeffs2: Distortion coefficients for cameras 1 and 2
# rvec1, tvec1, rvec2, tvec2: Rotation and translation vectors for cameras 1 and 2
# point1, point2: Corresponding 2D points in images 1 and 2
# initial_point: Initial estimate of the 3D point


# Function to compute reprojection error
def reprojection_error(params, K, dist_coeffs, rvec, tvec, point_2d):
    x, y, z = params
    point_3d = np.array([x, y, z])

    # Project the 3D point to 2D using the camera parameters
    projected_point, _ = cv2.projectPoints(np.array([point_3d]), rvec, tvec, K, dist_coeffs)

    # Compute the reprojection error
    error = point_2d - projected_point[0, 0]

    return error


def test_reprojection_optimisation():
    # Initial estimate of the 3D point
    # initial_point = np.array([x, y, z])

    # Bundle adjustment using least_squares
    result = None
    # result = least_squares(reprojection_error, initial_point,
    #                     args=(K1, dist_coeffs1, rvec1, tvec1, point1),
    #                     method='lm')

    # Refined 3D point coordinates
    refined_point = result.x

    # Print the refined point
    print("Refined 3D Point:", refined_point)