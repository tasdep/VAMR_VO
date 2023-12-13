import numpy as np
import cv2
import params.params as params
from scipy.spatial.distance import cdist
import math

from utils.state import State
from utils.image_processing import run_harris_detector, patch_describe_keypoints, triangulate_points_wrapper


def update_landmarks(
    state: State,
    img_prev: np.ndarray,
    img_new: np.ndarray,
    current_camera_pose: np.ndarray,
    K: np.ndarray,
    print_stats: bool = False,
) -> State:
    """
    Update the landmarks in the state by detecting and triangulating new keypoints.

    Parameters:
    - state: The current state containing landmarks and camera pose information.
    - img_prev: The image from the previous frame.
    - img_new: The image from the current frame.
    - current_camera_pose: 16X1 matrix pose of the camera in the current frame.
    - K: 3X3 camera intrinsic matrix.

    Returns:
    - The updated state with new landmarks.
    """
    new_C, new_descriptors = get_candidate_keypoints(state, img_new, print_stats=print_stats)
    state = get_updated_keypoints(state, current_camera_pose, img_prev, img_new, new_C, new_descriptors, print_stats=print_stats)
    state = triangulate_candidates(state, current_camera_pose, K, print_stats=print_stats)
    return state


def get_candidate_keypoints(state: State, img_new: np.ndarray, print_stats: bool = False) -> (np.ndarray, np.ndarray):
    """
    Get candidate keypoints in the new image that are not present in the state and
    generate descriptors.

    Parameters:
    - state: The current state containing landmarks and camera pose information.
    - img_new: The image from the current frame.
    - visualise: Flag to visualize the keypoints detection.
    - print_stats: Flag to print statistics about the keypoints detection.

    Returns:
    - new_C: Mx2 matrix of new candidate keypoints.
    - new_descriptors: Mx(descriptor length) matrix of descriptors
                        corresponding to the new candidate keypoints.
    """
    # get the new keypoints with harris detector
    keypoints_new = run_harris_detector(img_new)
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

    if print_stats:
        print(
            f"Returning {new_C.shape[0]} new candidate keypoints. {keypoints_new.shape[0]-new_C.shape[0]} were rejected as being to close to existing landmarks"
        )
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
    print_stats: bool = False,
) -> State:
    """
    Update the state with the new candidate keypoints detected in the current frame OR update
    previously tracked candidate keypoints.

    Parameters:
    - state: The current state containing landmarks and camera pose information.
    - current_camera_pose: 16X1 pose of the camera in the current frame.
    - img_prev: The image from the previous frame.
    - img_new: The image from the current frame.
    - new_C: MX2 matrix of new candidate keypoints.
    - new_descriptors: MX(descriptor length) matrix of descriptors
                        corresponding to the new candidate keypoints.

    Returns:
    - The updated state with new candidate keypoints.
    """
    updated_counter: int = 0
    # F,T will be the same size as new_C
    new_F = np.ndarray(new_C.shape)
    new_T = np.ndarray((new_C.shape[0], 16))
    old_descriptors = patch_describe_keypoints(img_prev, state.C.T, params.DESC_PATCH_RAD)
    # find best match for each new descriptor
    bf: cv2.BFMatcher = cv2.BFMatcher.create(cv2.NORM_L2)
    matches = bf.match(queryDescriptors=new_descriptors.astype(np.float32), trainDescriptors=old_descriptors.astype(np.float32))
    # note: matches is the length of query descriptors and in that order
    # check matches whether they are close enough to be the same
    for i, candidate in enumerate(new_C):
        try:
            m = matches[i]
        except IndexError:
            m = None
        if m and m.distance < params.MATCH_DISTANCE_THRESHOLD:
            updated_counter += 1
            # match => this is an existing candidate, propagate F,T entries
            new_F[i] = state.F.T[m.trainIdx]
            new_T[i] = state.T.T[m.trainIdx]
        else:
            # No match => this is a new candidate, create new F,T entries
            new_F[i] = new_C[i]
            new_T[i] = current_camera_pose

    if print_stats:
        print(f"{updated_counter}/{new_C.shape[0]} candidates were updated, {new_C.shape[0]-updated_counter} were added new.")
    # transpose to make them 2 X N
    state.update_candidates(new_C.T, new_F.T, new_T.T)
    return state


def triangulate_candidates(old_state: State, current_camera_pose: np.ndarray, K: np.ndarray, print_stats: bool = False) -> State:
    """
    Triangulate candidate keypoints where the angle between frames is greater than a threshold
    and update the state with the triangulated landmarks.

    Parameters:
    - old_state: The current state containing landmarks and camera pose information.
    - current_camera_pose: 16X1 pose of the camera in the current frame.
    - K: 3X3 camera intrinsic matrix.

    Returns:
    - The updated state with newly triangulated landmarks added.
    """
    current_landmarks: int = old_state.P.shape[1]
    if current_landmarks >= params.NUM_LANDMARKS_GOAL:
        if print_stats:
            print(
                f"Number of landmarks currently tracked is {old_state.P.shape[1]}/{params.NUM_LANDMARKS_GOAL}. Not adding more."
            )
        return old_state
    else:
        num_to_add: int = params.NUM_LANDMARKS_GOAL - current_landmarks

    angles: np.ndarray = np.zeros((old_state.C.shape[1]))
    # do projections
    for i, (C, F, T) in enumerate(zip(old_state.C.T, old_state.F.T, old_state.T.T)):
        v_orig = unit_vector_to_pixel_in_world(K, T.reshape((4, 4)), F)
        v_curr = unit_vector_to_pixel_in_world(K, current_camera_pose.reshape((4, 4)), C)
        angles[i] = angle_between_units(v_orig, v_curr)

    # assumption: larger angle => better landmark to start tracking
    # take the num_to_add largest angles, set the rest to zero
    small_indices = np.argsort(angles)[: (angles.shape[0] - num_to_add)]
    angles[small_indices] = 0
    # now filter to make sure we only have valid ones
    mask = np.rad2deg(angles) > params.TRIANGULATION_ANGLE_THRESHOLD

    # get updated C, F, T with ones that don't meet the threshold
    new_C = old_state.C[:, ~mask]
    new_F = old_state.F[:, ~mask]
    new_T = old_state.T[:, ~mask]

    # get the to triangulate points
    tri_C = old_state.C[:, mask]
    tri_F = old_state.F[:, mask]
    tri_T = old_state.T[:, mask]

    # those above threshold calculate X
    for i, (C, F, T) in enumerate(zip(tri_C.T, tri_F.T, tri_T.T)):
        new_X, mask= triangulate_points_wrapper(T.reshape(4, 4), current_camera_pose.reshape(4, 4), K, F, C)
        if mask.all():
            old_state.add_landmark(C.reshape(2, -1), new_X.reshape(3, -1))

    if print_stats:
        print(f"Of {old_state.C.shape[1]} candidates, {tri_C.shape[1]} were triangulated and added to state.(X/P)")
        print(f"Number of landmarks now tracked is {old_state.P.shape[1]}/{params.NUM_LANDMARKS_GOAL}.")

    # TODO refine X estimate with non linear optimisation
    # move candidate to X,P in state
    old_state.update_candidates(new_C, new_F, new_T)
    return old_state


def unit_vector_to_pixel_in_world(K: np.ndarray, T: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    """
    Convert pixel coordinates to unit vector in world coordinates.

    Parameters:
    - K:  3X3 camera intrinsic matrix
    - T: (3/4)X4 camera pose transform matrix
    - pixel: 2X1 pixel coordinates in the image

    Returns:
    - Unit vector in world coordinates.
    """
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
    """
    Calculate the angle between two unit vectors. Both in n dimensions.

    Parameters:
    - v1: First unit vector.
    - v2: Second unit vector.

    Returns:
    - Angle between the two unit vectors in radians.
    """
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
