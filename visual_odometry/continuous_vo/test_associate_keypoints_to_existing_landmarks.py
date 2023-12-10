import numpy as np
import cv2
from visual_odometry. utils import state
from continuous_vo import associate_keypoints_to_existing_landmarks as assoc

# To run, from visual_odometry directory:
# python -m continuous_vo.test_associate_keypoints_to_existing_landmarks

# Load test data.
keypoints = np.loadtxt("../local_data/test_data/keypoints.txt").T
p_W_landmarks = np.loadtxt("../local_data/test_data/p_W_landmarks.txt").T
prev_image = cv2.imread("../local_data/test_data/000000.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.imread("../local_data/test_data/000001.png", cv2.IMREAD_GRAYSCALE)
# Colour version so we can plot coloured tracking traces.
vis_image = cv2.imread("../local_data/test_data/000001.png", cv2.IMREAD_COLOR)

state = state.State()

state.update_landmarks(p_W_landmarks, keypoints)

for i in range(30):
    assoc.track_and_update(state, prev_image, new_image, True, vis_image)
    prev_image = new_image
    idx = "00000" + str(i + 2)
    new_image = cv2.imread(
        f"../local_data/test_data/{idx[-6:]}.png", cv2.IMREAD_GRAYSCALE
    )
    vis_image = cv2.imread(f"../local_data/test_data/{idx[-6:]}.png", cv2.IMREAD_COLOR)
