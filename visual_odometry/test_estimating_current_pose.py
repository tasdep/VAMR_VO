import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import state
from continuous_vo.estimating_current_pose import estimating_current_pose
from continuous_vo import associate_keypoints_to_existing_landmarks as assoc

# To run, from visual_odometry directory:
# #python3 -m continuous_vo.test_estimating_current_pose.py

# Load test data.
keypoints = np.loadtxt("visual_odometry/data/keypoints.txt").T
p_W_landmarks = np.loadtxt("visual_odometry/data/p_W_landmarks.txt").T
prev_image = cv2.imread("visual_odometry/data/000000.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.imread("visual_odometry/data/000001.png", cv2.IMREAD_GRAYSCALE)
# Colour version so we can plot coloured tracking traces.
vis_image = cv2.imread("visual_odometry/data/000001.png", cv2.IMREAD_COLOR)
K = np.loadtxt("visual_odometry/data/K.txt")

state = state.State()

state.update_landmarks(p_W_landmarks, keypoints)

fig = plt.figure()

for i in range(8):
    assoc.track_and_update(state, prev_image, new_image, False, vis_image)
    prev_image = new_image
    idx = "00000" + str(i + 2)
    new_image = cv2.imread(
        f"visual_odometry/data//{idx[-6:]}.png", cv2.IMREAD_GRAYSCALE
    )
    vis_image = cv2.imread(f"visual_odometry/data//{idx[-6:]}.png", cv2.IMREAD_COLOR)

    R, t = estimating_current_pose(state=state, K=K, PnP_solver='P3P', refine_with_DLT=False, visualization=True, figure=fig)