import numpy as np
import cv2
import matplotlib.pyplot as plt

import params.params as params
from utils.state import State


# TODO check if K is allowed to be used
# ASSUMPTION K_1 = K_2
def initialize_pipeline(input_images: np.ndarray, K: np.ndarray) -> State:
    # 2 frames from dataset and params
    # TODO read relevant images using data from params
    img1: np.ndarray = input_images[params.BOOTSRAP_FRAMES[0], ...]
    img2: np.ndarray = input_images[params.BOOTSRAP_FRAMES[1], ...]

    print(f"{img1.shape=}")
    assert img1.shape == img2.shape

    # find keypoint correspondences between frames, option to use intermediate frames
    # TODO patch matching to return list of keypoints for each image, harris/shi
    harris_params = {
        "blockSize": params.HARRIS_BLOCK_SIZE,
        "ksize": params.HARRIS_SOBEL_SIZE,
        "k": params.HARRIS_K,
    }
    corners_1: np.ndarray = cv2.cornerHarris(img1, **harris_params)
    corners_2: np.ndarray = cv2.cornerHarris(img2, **harris_params)

    print(f"{corners_1.shape=}")
    print(f"{corners_2.shape=}")

    # TODO threshold corner response
    keypoints_1 = np.argwhere(corners_1 > params.KEYPOINT_THRESHOLD * corners_1.max())
    # TODO
    keypoints_2 = np.argwhere(corners_2 > params.KEYPOINT_THRESHOLD * corners_2.max())

    print(f"{keypoints_1.shape=}")
    print(f"{keypoints_2.shape=}")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1, cmap="gray")
    axs[0].plot(keypoints_1[:, 1], keypoints_1[:, 0], "rx")
    axs[1].imshow(img2, cmap="gray")
    axs[1].plot(keypoints_2[:, 1], keypoints_2[:, 0], "rx")
    # axs[1].imshow(corners_1)
    plt.show()

    # TODO try adaptive thresholding, https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

    # TODO non maxima suppresion to select best n keypoints or just to reduce any crossover

    # TODO calc descriptors
    descriptors_1: np.ndarray = describe_keypoints(img1, keypoints_1, params.DESC_PATCH_RAD)
    descriptors_2: np.ndarray = describe_keypoints(img2, keypoints_2, params.DESC_PATCH_RAD)

    print(f"{descriptors_1.shape=}")
    print(f"{descriptors_2.shape=}")

    # TODO match based on descriptors, use scipy cdist to calc SSD or cv2 BF matcher
    # TODO try ratio test instead of crossCheck
    bf: cv2.BFMatcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    # TODO does this order matter
    # returns the nearest match for each, no outlier checking
    # query, train for input descriptors
    matches = bf.match(queryDescriptors=descriptors_1.astype(np.float32), trainDescriptors=descriptors_2.astype(np.float32))
    print(f"{len(matches)=}")

    # Create keypoint objects from numpy arrays
    # int to float casting was not working automatically
    keypoints1_lst: list[cv2.KeyPoint] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints_1]
    keypoints2_lst: list[cv2.KeyPoint] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints_2]

    # visualise matches
    img3 = cv2.drawMatches(img1, keypoints1_lst, img2, keypoints2_lst, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    # TODO performance dependent change to KLT algorithm

    # estimate relative pose between frames and triangulate 3D landmarks
    # TODO estimate fundamental matrix
    # get matched keypoints to estimate fundamental matrix
    matched_pts_1 = np.float32([keypoints_1[m.queryIdx] for m in matches]).reshape(-1, 2)
    matched_pts_2 = np.float32([keypoints_2[m.trainIdx] for m in matches]).reshape(-1, 2)

    fundamental: np.ndarray
    mask: np.ndarray
    fundamental, mask = cv2.findFundamentalMat(
        matched_pts_1, matched_pts_2, cv2.FM_RANSAC, params.RANSAC_REPROJ_THRESH, params.RANSAC_CONFIDENCE
    )
    print(f"{fundamental=}")
    print(f"{mask.sum()=}")

    # visualise inlier matches after RANSAC
    inlier_matches = [match for i, match in enumerate(matches) if mask[i] == 1]
    print(f"{len(matches)=}")
    print(f"{len(inlier_matches)=}")
    img3 = cv2.drawMatches(
        img1, keypoints1_lst, img2, keypoints2_lst, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.imshow(img3)
    plt.show()

    # TODO extract relative camera positions, R t
    # find essential matrix
    essential: np.ndarray = K.T @ fundamental @ K
    print(f"{essential=}")
    inlier_pts_1: np.ndarray = matched_pts_1[mask.ravel() == 1]
    inlier_pts_2: np.ndarray = matched_pts_2[mask.ravel() == 1]
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
    print(f"{R1=}")
    print(f"{np.linalg.det(R1)=}")
    print(f"{R2=}")
    print(f"{np.linalg.det(R2)=}")
    print(f"{t=}")

    # find correct combination out of the 4 possible options
    R_correct, t_correct = disambiguateRelativePose([R1, R2], t, inlier_pts_1, inlier_pts_2, K)

    # TODO what should the keypoints be here, first frame or second? I think second. Assume just filtered ones after running RANSAC
    # TODO triangulate point cloud
    # 3D
    X: np.ndarray = triangulate_points_wrapper(R_correct, t_correct, K, inlier_pts_1, inlier_pts_2)
    # 2D
    P: np.ndarray = inlier_pts_2.T
    # TODO does anything else need to be returned for pipeline?
    # - pose between frames
    # - estimated camera calibration

    # initialise the pipeline object with the keypoints and landmarks
    # note: C,F,T are blank as there are no candidates at this point
    return State().update_landmarks(X, P)


def describe_keypoints(img: np.ndarray, keypoints: np.ndarray, r: int) -> np.ndarray:
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    N: int = keypoints.shape[0]
    descriptors: np.ndarray = np.zeros([N, (2 * r + 1) ** 2])
    padded: np.ndarray = np.pad(img, [(r, r), (r, r)], mode="constant", constant_values=0)

    for i in range(N):
        kp: np.ndarray = keypoints[i, :].astype(int) + r
        descriptors[i, :] = padded[(kp[0] - r) : (kp[0] + r + 1), (kp[1] - r) : (kp[1] + r + 1)].flatten()

    return descriptors


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
