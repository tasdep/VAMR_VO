import numpy as np
import cv2
from utils.state import State
from continuous_vo.update_landmarks import get_candidate_keypoints, get_updated_keypoints, triangulate_candidates


# Load test data.
prev_image = cv2.imread("..\shared_data\parking\images\img_00000.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.imread("..\shared_data\parking\images\img_00000.png", cv2.IMREAD_GRAYSCALE)
# Colour version so we can plot coloured tracking traces.
vis_image = cv2.imread("..\shared_data\parking\images\img_00000.png", cv2.IMREAD_COLOR)

state = State()


new_C, new_descriptors = get_candidate_keypoints(state, new_image, print_stats=True, visualise=True)
state = get_updated_keypoints(state, np.eye(4).flatten(), prev_image, new_image, new_C, new_descriptors)
state = triangulate_candidates(state)
