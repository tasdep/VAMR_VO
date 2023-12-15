import cv2
from enum import Enum


class Dataset(Enum):
    DATASET1 = "Shared dataset"
    DATASET2 = "Kitti"
    DATASET3 = "Malaga"
    DATASET4 = "Ransac exercise"


DATASET = Dataset.DATASET2
# Which image to begin on in the dataset
START_IMG_IDX = 155

# turning on profiling disables the visualiser
# output is a file 'full_run.stats'
# to visualise use cmd line tool snakeviz "snakeviz *.stats"
DO_PROFILING = False

# wait for arrow key to advance to next frame
WAIT_ARROW = True

# Whether to visualize the global point cloud or just the
# actively tracked point cloud.
GLOBAL_POINT_CLOUD = False


# limit the number of frames
LIMIT_FRAME_COUNT = True
FRAME_LIMIT = 1000

################################################################
# Params for 3 - Initialization #
################################################################

# Whether to use the skip and use the test data that already has
# correspondences. In order to use this, you must copy the dataset
# from the RANSAC class exercise into your local_data folder.
SKIP_BOOTSTRAP = False

# Which frames from the input to use for bootstrapping initial features.
BOOTSRAP_FRAMES = [0, 3]

HARRIS_BLOCK_SIZE = 3
HARRIS_MAX_CORNERS = 1000
HARRIS_MIN_DISTANCE = 10.0
HARRIS_QUALITY_LEVEL = 0.01

# after harris corner detector to threshold which points are corners
KEYPOINT_THRESHOLD = 0.2

DESC_PATCH_RAD = 3

RANSAC_REPROJ_THRESH = 0.5
RANSAC_CONFIDENCE = 0.999

OUTLIER_3D_REJECTION_SIGMA = 3.0


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
    20,  # Maximum number of iterations the algorithm will take
    0.003,  # Epsilon value for convergence
)

TRACKING_OUTLIER_REJECTION_SIGMA = 2.0

# Params for 4.2 - RANSAC localization for pose estimation #
################################################################

# Number of iterations. Default: 100
POSE_RANSAC_ITERATION = 2000

# Inlier threshold value used by the RANSAC procedure.
# The parameter value is the maximum allowed distance between the observed and computed point projections to consider it an inlier. Default: 8.0
POSE_RANSAC_REPROJECTION_ERROR = 10.0

# The probability that the algorithm produces a useful result. Default: 0.99
POSE_RANSAC_CONFIDENCE = 0.999

################################################################
# Params for 4.3 - Associating keypoints to exisitng landmarks #
################################################################
# threshold to determine whether a newly detected keypoint is the same as a currently tracked one
# eg. when comparing candidate keypoints to state.P
# value is a pixel radius
EQUAL_KEYPOINT_THRESHOLD = 8.0

# minimum 'distance' between matches for them to be equal
MATCH_DISTANCE_THRESHOLD = 300

# threshold for angle between camera poses to add candidate to landmark set
TRIANGULATION_ANGLE_THRESHOLD = 5  # [deg]

# number of landmarks to maintain
NUM_LANDMARKS_GOAL = 500

LIMIT_NEW_LANDMARKS = 100
