import numpy as np 


import params.params as params
from utils.state import State


# TODO work out input output
def initialize_pipeline(images: np.ndarray) -> State:
    # 2 frames from dataset and params

    # find keypoint correspondences between frames, option to use intermediate frames

    # estimate relative pose between frames and triangulate 3D landmarks
    X: np.ndarray = None
    P: np.ndarray = None

    # initialise the pipeline object with the keypoints and landmarks
    # note: C,F,T are blank as there are no candidates at this point
    return State().update_landmarks(X, P)