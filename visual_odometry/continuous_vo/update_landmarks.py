import numpy as np
import cv2
import matplotlib.pyplot as plt
import params.params as params
import math
from scipy.spatial.distance import cdist

from utils.state import State
from utils.image_processing import run_harris_detector, patch_describe_keypoints, triangulate_points_wrapper


def update_landmarks(
    state: State, img_prev: np.ndarray, img_new: np.ndarray, current_camera_pose: np.ndarray, K: np.ndarray
) -> State:
    new_C, new_descriptors = get_candidate_keypoints(state, img_new)
    state = get_updated_keypoints(state, current_camera_pose, img_prev, img_new, new_C, new_descriptors)
    state = triangulate_candidates(state, current_camera_pose, K)
    return state


def get_candidate_keypoints(
    state: State, img_new: np.ndarray, visualise: bool = False, print_stats: bool = False
) -> (np.ndarray, np.ndarray):
    # get the new keypoints with harris detector
    keypoints_new = run_harris_detector(img_new, visualise, print_stats)
    new_C: np.ndarray
    new_descriptors: np.ndarray
    # compare each keypoint in P with the new kepoints
    # and only keep those that don't have a match in P

    if state.P is not None:
        # distances is an #keypoints_new x #state.P
        distances: np.ndarray = cdist(keypoints_new, state.P.T, metric="cityblock")
        # for each new point, find closest old eg. min along rows
        mins: np.ndarray = np.argmin(distances, axis=1)
        # create mask where min distance < params.EQUAL_KEYPOINT_THRESHOLD
        mask: np.ndarray = mins > params.EQUAL_KEYPOINT_THRESHOLD

        new_C = keypoints_new[mask, :]

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
def triangulate_candidates(old_state: State, current_camera_pose: np.ndarray, K: np.ndarray) -> State:
    angles: np.ndarray = np.zeros((old_state.C.shape[1]))
    # do projections
    for i, (C, F, T) in enumerate(zip(old_state.C.T, old_state.F.T, old_state.T.T)):
        v_orig = unit_vector_to_pixel_in_world(K, T.reshape((4,4)), F)
        v_curr = unit_vector_to_pixel_in_world(K, current_camera_pose.reshape((4,4)), C)
        angles[i] = angle_between_units(v_orig, v_curr)

    mask = angles > params.TRIANGULATION_ANGLE_THRESHOLD

    # get updated C, F, T with ones that don't meet the threshold
    new_C = old_state.C[:,~mask]
    new_F = old_state.F[:,~mask]
    new_T = old_state.T[:,~mask]

    # get the to triangulate points
    tri_C = old_state.C[:,mask]
    tri_F = old_state.F[:,mask]
    tri_T = old_state.T[:,mask]

    # those above threshold calculate X
    for i, C, F, T in enumerate(zip(tri_C.T, tri_F.T, tri_T.T)):
        new_X = triangulate_points_wrapper(T, current_camera_pose, K, F, C)
        old_state.add_landmark(C, new_X)

    # TOODO refine X estimate with non linear optimisation
    # move candidate to X,P in state
    old_state.update_candidates(new_C, new_F, new_T)
    return old_state


def unit_vector_to_pixel_in_world(K: np.ndarray, T: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    # extract rot matrix
    R = T[0:3, 0:3]
    # Inverse of the camera matrix
    K_inv = np.linalg.inv(K)

    # Convert from pixel coordinates to camera coords,
    # vector from camera base to the point in camera coordinates
    ndc_homogeneous = np.hstack((pixel, 1))
    direction_camera_coordinates = np.dot(K_inv, ndc_homogeneous)

    # normalise the vector because we only care about direction
    vector_camera_coordinates = direction_camera_coordinates / np.linalg.norm(direction_camera_coordinates)

    # Transform to world coordinates
    vector_world_coordinates = np.dot(R.T, vector_camera_coordinates)

    return vector_world_coordinates


def angle_between_units(v1: np.ndarray, v2: np.ndarray):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
