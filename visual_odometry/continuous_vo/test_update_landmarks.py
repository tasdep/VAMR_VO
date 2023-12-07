import numpy as np
import cv2
from visual_odometry. utils import state
# from continuous_vo.update_landmarks import get_candidate_keypoints,get_updated_keypoints
from continuous_vo.update_landmarks import get_candidate_keypoints, get_updated_keypoints


# Load test data.
prev_image = cv2.imread("../local_data/test_data/000000.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.imread("../local_data/test_data/000001.png", cv2.IMREAD_GRAYSCALE)
# Colour version so we can plot coloured tracking traces.
vis_image = cv2.imread("../local_data/test_data/000001.png", cv2.IMREAD_COLOR)

state = state.State()


new_C, new_descriptors = get_candidate_keypoints(None,state,new_image)
state = get_updated_keypoints(state,None, prev_image,new_image, new_C, new_descriptors)
