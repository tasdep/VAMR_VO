import numpy as np
import cv2
import matplotlib.pyplot as plt


import params.params as params
from utils.visualisation import drawCamera
from utils.state import State
from utils.image_processing import run_harris_detector,describe_keypoints


# ASSUMPTION K_1 = K_2
def initialize_pipeline(input_images: np.ndarray, K: np.ndarray, visualise: bool = False, print_stats: bool = False) -> State:
    # 2 frames from dataset and params
    img_1: np.ndarray = input_images[params.BOOTSRAP_FRAMES[0], ...]
    img_2: np.ndarray = input_images[params.BOOTSRAP_FRAMES[1], ...]

    assert img_1.shape == img_2.shape

    # # find keypoint correspondences between frames, option to use intermediate frames
    # harris_params = {
    #     "blockSize": params.HARRIS_BLOCK_SIZE,
    #     "ksize": params.HARRIS_SOBEL_SIZE,
    #     "k": params.HARRIS_K,
    # }
    # corners_1: np.ndarray = cv2.cornerHarris(img_1, **harris_params)
    # corners_2: np.ndarray = cv2.cornerHarris(img_2, **harris_params)

    # # extract keypoints from corner detector
    # keypoints_1 = np.argwhere(corners_1 > params.KEYPOINT_THRESHOLD * corners_1.max())
    # keypoints_2 = np.argwhere(corners_2 > params.KEYPOINT_THRESHOLD * corners_2.max())

    # if print_stats:
    #     print(f"{keypoints_1.shape=}")
    #     print(f"{keypoints_2.shape=}")

    # if visualise:
    #     fig, axs = plt.subplots(1, 2)
    #     axs[0].imshow(img_1, cmap="gray")
    #     axs[0].plot(keypoints_1[:, 1], keypoints_1[:, 0], "rx")
    #     axs[1].imshow(img_2, cmap="gray")
    #     axs[1].plot(keypoints_2[:, 1], keypoints_2[:, 0], "rx")
    #     plt.show()

    keypoints_1 = run_harris_detector(img_1,visualise, print_stats)
    keypoints_2 = run_harris_detector(img_2,visualise, print_stats)


    # TODO try adaptive thresholding, https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html to acheive more uniform keypoints
    # TODO non maxima suppresion to select best n keypoints or just to reduce any crossover
    # TODO try KLT instead of descriptor matching

    # calculate patch descriptors
    descriptors_1: np.ndarray = describe_keypoints(img_1, keypoints_1, params.DESC_PATCH_RAD)
    descriptors_2: np.ndarray = describe_keypoints(img_2, keypoints_2, params.DESC_PATCH_RAD)

    # TODO try ratio test instead of crossCheck

    # match the descriptors between images
    # cross check only returns matches where the nearest keypoint is consistent between images
    bf: cv2.BFMatcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(queryDescriptors=descriptors_1.astype(np.float32), trainDescriptors=descriptors_2.astype(np.float32))

    if print_stats:
        print(f"{len(matches)=}")

    # Create keypoint objects from numpy arrays
    keypoints1_lst: list[cv2.KeyPoint] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints_1]
    keypoints2_lst: list[cv2.KeyPoint] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints_2]

    if visualise:
        img3 = cv2.drawMatches(
            img_1, keypoints1_lst, img_2, keypoints2_lst, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.imshow(img3)
        plt.show()

    # get matched keypoints to estimate fundamental matrix
    matched_pts_1 = np.float32([keypoints_1[m.queryIdx] for m in matches]).reshape(-1, 2)
    matched_pts_2 = np.float32([keypoints_2[m.trainIdx] for m in matches]).reshape(-1, 2)

    fundamental: np.ndarray
    mask: np.ndarray
    fundamental, mask = cv2.findFundamentalMat(
        matched_pts_1, matched_pts_2, cv2.FM_RANSAC, params.RANSAC_REPROJ_THRESH, params.RANSAC_CONFIDENCE
    )

    # visualise inlier matches after RANSAC
    inlier_matches = [match for i, match in enumerate(matches) if mask[i] == 1]
    if print_stats:
        print(f"{len(inlier_matches)=}")

    if visualise:
        img3 = cv2.drawMatches(
            img_1, keypoints1_lst, img_2, keypoints2_lst, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.imshow(img3)
        plt.show()

    # find essential matrix
    essential: np.ndarray = K.T @ fundamental @ K

    # get remaining keypoints after RANSAC
    inlier_pts_1: np.ndarray = matched_pts_1[mask.ravel() == 1]
    inlier_pts_2: np.ndarray = matched_pts_2[mask.ravel() == 1]
    if print_stats:
        print(f"{inlier_pts_1.shape=}")
        print(f"{inlier_pts_2.shape=}")

    # extract parts, 2 possible rotations and the translations could be negative => 4 combinations
    R1: np.ndarray
    R2: np.ndarray
    t: np.ndarray
    R1, R2, t = cv2.decomposeEssentialMat(essential)

    # fix determinants
    if np.linalg.det(R1) < 0:
        R1[:, 0] *= -1
    if np.linalg.det(R2) < 0:
        R2[:, 0] *= -1

    # find correct combination out of the 4 possible options
    R_correct, t_correct = disambiguateRelativePose([R1, R2], t, inlier_pts_1, inlier_pts_2, K)

    # 3D
    X: np.ndarray = triangulate_points_wrapper(R_correct, t_correct, K, inlier_pts_1, inlier_pts_2)
    # 2D
    P: np.ndarray = inlier_pts_2.T

    # Visualisation
    if visualise:
        visualise_pose_and_landmarks(R_correct, t_correct, img_1, img_2, X, inlier_pts_1, inlier_pts_2)

    # initialise the pipeline object with the keypoints and landmarks
    # note: C,F,T are blank as there are no candidates at this point
    return State().update_landmarks(X, P)




def disambiguateRelativePose(rots: np.ndarray, t: np.ndarray, points_1: np.ndarray, points_2: np.ndarray, K: np.ndarray):
    """DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
    four possible configurations) by returning the one that yields points
    lying in front of the image plane (with positive depth).

    Arguments:
      Rots -  list[3x3]: the two possible rotations returned by decomposeEssentialMatrix
      t   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
      p1   -  2xN coordinates of point correspondences in image 1
      p2   -  2xN coordinates of point correspondences in image 2
      K   -  3x3 calibration matrix for camera 1/2

    Returns:
      R -  3x3 the correct rotation matrix
      T -  3x1 the correct translation vector

      where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
      from the world coordinate system (identical to the coordinate system of camera 1)
      to camera 2.
    """

    # Projection matrix of camera 1
    M1: np.ndarray = K @ np.eye(3, 4)
    ts: list[np.ndarray] = [t, -t]
    total_points_in_front_best: int = 0
    for rot in rots:
        for t in ts:
            M2: np.ndarray = K @ np.c_[rot, t]
            P_C1: np.ndarray = cv2.triangulatePoints(M1, M2, points_1.T, points_2.T)

            # project in both cameras
            P_C2: np.ndarray = np.c_[rot, t] @ P_C1

            # dehomegnise
            P_C1: np.ndarray = P_C1[0:3, :] / P_C1[3, :]

            num_points_in_front1: int = np.sum(P_C1[2, :] > 0)
            num_points_in_front2: int = np.sum(P_C2[2, :] > 0)
            total_points_in_front: int = num_points_in_front1 + num_points_in_front2

            if total_points_in_front > total_points_in_front_best:
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R: np.ndarray = rot
                T: np.ndarray = t
                total_points_in_front_best = total_points_in_front

    return R, T


def triangulate_points_wrapper(rot: np.ndarray, t: np.ndarray, K: np.ndarray, points_1: np.ndarray, points_2: np.ndarray):
    M1: np.ndarray = K @ np.eye(3, 4)
    M2: np.ndarray = K @ np.c_[rot, t]
    points_3D: np.ndarray = cv2.triangulatePoints(M1, M2, points_1.T, points_2.T)
    # dehomgenize
    points_3D: np.ndarray = points_3D[0:3, :] / points_3D[3, :]
    return points_3D


def visualise_pose_and_landmarks(
    R: np.ndarray,
    t: np.ndarray,
    img_1: np.ndarray,
    img_2: np.ndarray,
    X: np.ndarray,
    inliers_1: np.ndarray,
    inliers_2: np.ndarray,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection="3d")

    # R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

    # P is a [4xN] matrix containing the triangulated point cloud (in
    # homogeneous coordinates), given by the function linearTriangulation
    ax.scatter(X[0, :], X[1, :], X[2, :], marker="o")
    # Display camera pose
    drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale=2)
    ax.text(-0.1, -0.1, -0.1, "Cam 1")

    center_cam2_W = -R.T @ t
    center_cam2_W = center_cam2_W.reshape(-1)
    drawCamera(ax, center_cam2_W, R.T, length_scale=2)
    ax.text(center_cam2_W[0] - 0.1, center_cam2_W[1] - 0.1, center_cam2_W[2] - 0.1, "Cam 2")

    # Display matched points
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(img_1)
    ax.scatter(inliers_1.T[1, :], inliers_1.T[0, :], color="y", marker="s")
    ax.set_title("Image 1")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img_2)
    ax.scatter(inliers_2.T[1, :], inliers_2.T[0, :], color="y", marker="s")
    ax.set_title("Image 2")

    plt.show()
