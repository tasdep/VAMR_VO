import cv2
import numpy as np
import matplotlib.pyplot as plt

import params.params as params

from bootstrapping.initialization import initialize_pipeline
from utils.state import State

###############################
# Load data                   #
###############################
test = np.array([1, 2, 3, 4, 5])
images: np.ndarray = np.zeros((10,100,100))
###############################
# Bootstrap Initial Landmarks #
###############################
initial_state: State = initialize_pipeline(images)

## Get 2D Point Correspondences
print(test[params.BOOTSRAP_FRAMES])

## Estimate relative pose and triangulate point cloud
## - 8 point algorithm with RANSAC

###############################
# Continuous Visual Odometry  #
###############################
