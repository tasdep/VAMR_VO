import numpy as np
import cv2
import matplotlib.pyplot as plt
import params.params as params
import math

from utils.state import State
from utils.image_processing import run_harris_detector, patch_describe_keypoints


def get_candidate_keypoints(
    state: State, img_new: np.ndarray, visualise: bool = False, print_stats: bool = False
) -> (np.ndarray, np.ndarray):
    # get the new keypoints with harris detector
    keypoints_new = run_harris_detector(img_new, visualise, print_stats)
    new_C: np.ndarray
    new_descriptors: np.ndarray
    # compare each keypoint in P with the new kepoints
    # and only keep those that don't have a match in P
    if state.P:
        for keypoint_new in keypoints_new:
            flag: bool = False
            for keypoint_old in state.P:
                # check similarity of location
                if np.allclose(keypoint_new, keypoint_old, atol=params.EQUAL_KEYPOINT_THRESHOLD, rtol=0):
                    flag = True
            if flag is False:
                # not similar so save it
                new_C.append(keypoint_new)
    else:
        new_C = keypoints_new

    # calculate the descriptors of the new keypoints
    new_descriptors = patch_describe_keypoints(img_new, new_C, params.DESC_PATCH_RAD)
    return new_C, new_descriptors


def get_updated_keypoints(
    state: State,
    current_camera_pose: np.ndarray,
    img_prev: np.ndarray,
    img_new: np.ndarray,
    new_C: np.ndarray,
    new_descriptors: np.ndarray,
) -> State:
    # F,T will be the same size as new_C
    new_F = np.ndarray(new_C.shape)
    new_T = np.ndarray((new_C.shape[0], 16))
    old_descriptors = patch_describe_keypoints(img_prev, state.C, params.DESC_PATCH_RAD)

    # find best match for each new descriptor
    bf: cv2.BFMatcher = cv2.BFMatcher.create(cv2.NORM_L2)
    matches = bf.match(queryDescriptors=old_descriptors.astype(np.float32), trainDescriptors=new_descriptors.astype(np.float32))

    # check matches whether they are close enough to be the same
    for i, candidate in enumerate(new_C):
        try:
            m = matches[i]
        except IndexError:
            m = None
        if m and m.distance < params.MATCH_DISTANCE_THRESHOLD:
            # match => this is an existing candidate, propagate F,T entries
            new_F[i] = state.F[m.queryIdx]
            new_T[i] = state.T[m.queryIdx]
        else:
            # No match => this is a new candidate, create new F,T entries
            new_F[i] = new_C[i]
            new_T[i] = current_camera_pose

    # transpose to make them 2 X N
    state.update_candidates(new_C.T, new_F.T, new_T.T)
    return state


# TODO to be run after updating candidates
# this checks if there are any candidates for triangulation and moves them from C,F,T to X,P
def triangulate_candidates(old_state: State, current_camera_pose: np.ndarray) -> State:
    # do projections
    for C,F,T in zip(old_state.C, old_state.F, old_state.T):
        print(C,F,T)
    # calculate angle for each candiate
    # threshold angles
    # those above threshold calculate X 
    # refine X estimate with solvePnP
    # move candidate to X,P in state
    return old_state
