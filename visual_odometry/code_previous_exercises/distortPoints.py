import numpy as np


def distortPoints(x, D, K):
    """Applies lens distortion D(2) to 2D points x(Nx2) on the image plane. """

    k1, k2 = D[0], D[1]

    u0 = K[0, 2]
    v0 = K[1, 2]

    xp = x[:, 0] - u0
    yp = x[:, 1] - v0

    r2 = xp**2 + yp**2
    xpp = u0 + xp * (1 + k1*r2 + k2*r2**2)
    ypp = v0 + yp * (1 + k1*r2 + k2*r2**2)

    x_d = np.stack([xpp, ypp], axis=-1)

    return x_d
