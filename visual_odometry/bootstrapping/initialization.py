import numpy as np
import cv2
import matplotlib.pyplot as plt

import params.params as params
from utils.state import State


# TODO work out input output
def initialize_pipeline(input_images: np.ndarray) -> State:
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
    corners_2: np.ndarray = cv2.cornerHarris(img1, **harris_params)

    print(f"{corners_1.shape=}")
    print(f"{corners_2.shape=}")

    # TODO threshold corner response
    keypoints1 = np.argwhere(corners_1 > params.KEYPOINT_THRESHOLD * corners_1.max())
    keypoints2 = np.argwhere(corners_2 > params.KEYPOINT_THRESHOLD * corners_2.max())

    print(f"{keypoints1.shape=}")
    print(f"{keypoints2.shape=}")

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1, cmap="gray")
    axs[0].plot(keypoints1[:, 1], keypoints1[:, 0], "rx")
    axs[1].imshow(corners_1)
    plt.show()

    # TODO try adaptive thresholding, https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

    # TODO non maxima suppresion to select best n keypoints or just to reduce any crossover

    # TODO calc descriptors
    descriptors1: np.ndarray = describe_keypoints(img1, keypoints1, params.DESC_PATCH_RAD)
    descriptors2: np.ndarray = describe_keypoints(img2, keypoints2, params.DESC_PATCH_RAD)

    print(f"{descriptors1.shape=}")
    print(f"{descriptors2.shape=}")

    # TODO match based on descriptors, use scipy cdist to calc SSD or cv2 BF matcher
    # TODO try ratio test instead of crossCheck
    bf: cv2.BFMatcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)

    # TODO does this order matter
    # returns the nearest match for each, no outlier checking
    matches = bf.match(descriptors2.astype(np.float32), descriptors1.astype(np.float32))
    print(f"{len(matches)=}")

    # Create keypoint objects from numpy arrays
    # int to float casting was not working automatically
    keypoints1_lst: list[cv2.KeyPoint] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints1]
    keypoints2_lst: list[cv2.KeyPoint] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in keypoints2]

    # visualise matches
    img3 = cv2.drawMatches(img1, keypoints1_lst, img2, keypoints2_lst, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    # TODO performance dependent change to KLT algorithm

    # estimate relative pose between frames and triangulate 3D landmarks
    X: np.ndarray = None
    P: np.ndarray = None
    # TODO estimate essential matrix
    # TODO extract relative camera positions
    # TODO triangulate point cloud

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
