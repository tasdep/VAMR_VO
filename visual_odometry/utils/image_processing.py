import numpy as np
import cv2
import matplotlib.pyplot as plt
import params.params as params


def run_harris_detector(img: np.ndarray, visualise: bool = False, print_stats: bool = False):
    # find keypoint correspondences between frames, option to use intermediate frames
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
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
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
