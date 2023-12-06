import numpy as np
import cv2
from utils import state
from continuous_vo import estimating_current_pose

# To run, from visual_odometry directory:
# #python3 -m continuous_vo.test_estimating_current_pose.py

# Load test data.
keypoints = np.loadtxt("VAMR_VO/visual_odometry/data/keypoints.txt").T
p_W_landmarks = np.loadtxt("VAMR_VO/visual_odometry/data/p_W_landmarks.txt").T

K = np.loadtxt("VAMR_VO/visual_odometry/data/K.txt")

state = state.State()

state.update_landmarks(p_W_landmarks, keypoints)

T = estimating_current_pose(state, K)