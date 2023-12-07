import numpy as np
import cv2
import matplotlib.pyplot as plt
import params.params as params


def run_harris_detector(img_1: np.ndarray, visualise: bool = False, print_stats: bool = False):

   # find keypoint correspondences between frames, option to use intermediate frames
    harris_params = {
        "blockSize": params.HARRIS_BLOCK_SIZE,
        "ksize": params.HARRIS_SOBEL_SIZE,
        "k": params.HARRIS_K,
    }
    corners_1: np.ndarray = cv2.cornerHarris(img_1, **harris_params)

    # extract keypoints from corner detector
    keypoints_1 = np.argwhere(corners_1 > params.KEYPOINT_THRESHOLD * corners_1.max())

    if print_stats:
        print(f"{keypoints_1.shape=}")

    if visualise:
        fig, axs = plt.subplots(1, 1)
        axs[0].imshow(img_1, cmap="gray")
        axs[0].plot(keypoints_1[:, 1], keypoints_1[:, 0], "rx")
        plt.show()

    return keypoints_1

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
