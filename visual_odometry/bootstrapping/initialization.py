import numpy as np 


import params.params as params
from utils.state import State


# TODO work out input output
def initialize_pipeline(images: np.ndarray) -> State:
    # 2 frames from dataset and params
    # TODO read relevant images using data from params


    # find keypoint correspondences between frames, option to use intermediate frames
    # TODO patch matching to return list of keypoint pairs in 2D
    # TODO performance dependent change to KLT algorithm 

    # estimate relative pose between frames and triangulate 3D landmarks
    X: np.ndarray = None
    P: np.ndarray = None
    # TODO estimate essential matrix
    # TODO extract relative camera positions
    # TODO triangulate point cloud

    # initialise the pipeline object with the keypoints and landmarks
    # note: C,F,T are blank as there are no candidates at this point
    return State().update_landmarks(X, P)