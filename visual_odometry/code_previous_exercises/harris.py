import numpy as np
from scipy import signal


def harris(img, patch_size, kappa):
    sobel_para = np.array([-1, 0, 1])
    sobel_orth = np.array([1, 2, 1])

    Ix = signal.convolve2d(img, sobel_para[None, :], mode="valid")
    Ix = signal.convolve2d(Ix, sobel_orth[:, None], mode="valid").astype(float)

    Iy = signal.convolve2d(img, sobel_para[:, None], mode="valid")
    Iy = signal.convolve2d(Iy, sobel_orth[None, :], mode="valid").astype(float)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix*Iy

    patch = np.ones([patch_size, patch_size])
    pr = patch_size // 2
    sIxx = signal.convolve2d(Ixx, patch, mode="valid")
    sIyy = signal.convolve2d(Iyy, patch, mode="valid")
    sIxy = signal.convolve2d(Ixy, patch, mode="valid")

    scores = (sIxx * sIyy - sIxy ** 2) - kappa * ((sIxx + sIyy) ** 2)

    scores[scores < 0] = 0

    scores = np.pad(scores, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    return scores

