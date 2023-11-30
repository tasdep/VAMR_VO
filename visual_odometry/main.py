import cv2
import numpy as np
import matplotlib.pyplot as plt

import params.params as params

###############################
# Load data                   #
###############################
test = np.array([1, 2, 3, 4, 5])

###############################
# Bootstrap Initial Landmarks #
###############################

## Get 2D Point Correspondences
print(test[params.BOOTSRAP_FRAMES])

## Estimate relative pose and triangulate point cloud
## - 8 point algorithm with RANSAC

###############################
# Continuous Visual Odometry  #
###############################
