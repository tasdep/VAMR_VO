import cv2

################################################################
# Params for 3 - Initialization #
################################################################
# Which frames from the input to use for bootstrapping initial features.
BOOTSRAP_FRAMES = [0, 3]

HARRIS_BLOCK_SIZE = 9
HARRIS_SOBEL_SIZE = 3
HARRIS_K = 0.1

KEYPOINT_THRESHOLD = 0.01

DESC_PATCH_RAD = 3

RANSAC_REPROJ_THRESH = 0.5
RANSAC_CONFIDENCE = 0.999


################################################################
# Params for 4.1 - Associating keypoints to exisitng landmarks #
################################################################
KLT_WINDOW_SIZE = (15, 15)
# How many layers of downsampling to use to allow for more motion between frames.
KLT_MAX_LEVEL = 2
# TERM_CRITERIA_EPS terminates after the keypoint movement is less than epsilon, TERM_CRITERIA_COUNT terminates
# after a set number of iterations. Oring them results in termination after whichever criteria is met first.
KLT_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  # Criteria types: EPS or COUNT
    10,  # Maximum number of iterations the algorithm will take
    0.03,  # Epsilon value for convergence
)

################################################################
# Params for 4.2 - RANSAC localization for pose estimation #
################################################################

# Number of iterations. Default: 100
POSE_RANSAC_ITERATION = 2000

# Inlier threshold value used by the RANSAC procedure. 
# The parameter value is the maximum allowed distance between the observed and computed point projections to consider it an inlier. Default: 8.0
POSE_RANSAC_REPROJECTION_ERROR = 20.0

# The probability that the algorithm produces a useful result. Default: 0.99
POSE_RANSAC_CONFIDENCE = 0.999

