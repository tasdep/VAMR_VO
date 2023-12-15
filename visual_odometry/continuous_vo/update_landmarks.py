import numpy as np
import cv2
import params.params as params
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt


from utils.state import State
from utils.image_processing import (
    run_harris_detector,
    patch_describe_keypoints,
    triangulate_points_wrapper,
)


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
    new_keypoints = get_candidate_keypoints(state, img_new, print_stats=print_stats)
    state = get_updated_keypoints(
        state,
        current_camera_pose,
        img_prev,
        img_new,
        new_keypoints,
        print_stats=print_stats,
    )
    state = triangulate_candidates(
        state, current_camera_pose, K, print_stats=print_stats
    )
    return state


def get_candidate_keypoints(
    state: State, img_new: np.ndarray, print_stats: bool = False
) -> np.ndarray:
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
    """
    # get the new keypoints with harris detector
    # keypoints_new = run_harris_detector(img_new)
    sift = cv2.SIFT.create(nfeatures=500)
    sift_keypoints = sift.detect(img_new)
    keypoints_new = cv2.KeyPoint.convert(sift_keypoints)
    new_C: np.ndarray

    # compare each keypoint in P with the new kepoints
    # and only keep those that don't have a match in P
    if state.P is not None:
        # distances is an #keypoints_new x #state.P
        distances: np.ndarray = cdist(keypoints_new, state.P.T, metric="cityblock")
        # for each new point, find closest old eg. min along rows
        mins: np.ndarray = np.min(distances, axis=1)
        # create mask where min distance > params.EQUAL_KEYPOINT_THRESHOLD
        mask: np.ndarray = mins > params.EQUAL_KEYPOINT_THRESHOLD

        new_C = keypoints_new[mask, :]

    else:
        new_C = keypoints_new

    if print_stats:
        print(
            f"UPDATE LANDMARKS: Returning {new_C.shape[0]} new candidate keypoints. {keypoints_new.shape[0]-new_C.shape[0]} were rejected as being to close to existing landmarks"
        )
    return new_C


def get_updated_keypoints(
    state: State,
    current_camera_pose: np.ndarray,
    img_prev: np.ndarray,
    img_new: np.ndarray,
    new_keypoints: np.ndarray,
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
    - new_keypoints: MX2 matrix of new candidate keypoints.

    Returns:
    - The updated state with new candidate keypoints.
    """
    updated_counter: int = 0

    # create array of indices to keep track of which points in state.C
    # are tracked through KLT and RANSAC
    # mask this after each step
    indices_old_C: np.ndarray = np.arange(state.C.shape[1])

    # check if we have existing candidates
    if state.C.size > 0:
        # use KLT to track old candidate keypoints, output is predicted new keypoints
        # Parameters for KLT tracker
        lk_params = {
            "winSize": params.KLT_WINDOW_SIZE,
            "maxLevel": params.KLT_MAX_LEVEL,
            "criteria": params.KLT_CRITERIA,
        }
        # Track the keypoints in the new image
        # new_keypoints.shape = Nx1x2
        old_C = state.C.T.reshape(-1, 1, 2)
        old_C = np.array(old_C, dtype=np.float32)
        KLT_new_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_new, old_C, None, **lk_params
        )

        # Filter out the keypoints for which tracking was successful
        KLT_tracked_new = KLT_new_keypoints[status == 1]  # Shape: Nx2
        KLT_tracked_prev = old_C[status == 1]  # Shape: Nx2
        indices_old_C = indices_old_C[status.ravel() == 1]

        # RANSAC tracked points to reduce error
        _, inlier_mask = cv2.findFundamentalMat(
            KLT_tracked_prev, KLT_tracked_new, cv2.FM_RANSAC
        )

        # filter out points after RANSAC, this leaves points that were succesfully tracked
        # between frames
        RANSAC_inlier_new = KLT_tracked_new[inlier_mask.ravel() == 1]  # Shape Nx2
        RANSAC_inlier_prev = KLT_tracked_prev[inlier_mask.ravel() == 1]  # Shape Nx2
        indices_old_C = indices_old_C[inlier_mask.ravel() == 1]

        ################# SUCCESSFULLY TRACKED OLD POINTS
        # find successfully tracked points that match with existing and propogate state
        tracked_C = RANSAC_inlier_new  # Shape Nx2
        tracked_F = state.F.T[indices_old_C, :]  # Shape Nx2
        tracked_T = state.T.T[indices_old_C, :]  # Shape Nx2

        ################# NEWLY DETECTED CANDIDATES THAT DON'T MATCH THE TRACKED POINTS
        # append all other newly found keypoints with current camera pose to state
        # compare the RANSAC_inlier_new 2D points by location to new_keypoints and only keep those which don't overlap
        # these will form the newly tracked candidates
        # distances is an size(arg1) x size(arg2) array
        distances: np.ndarray = cdist(
            new_keypoints, RANSAC_inlier_new, metric="cityblock"
        )
        # for each new point, find closest old eg. min along rows
        mins: np.ndarray = np.min(distances, axis=1)
        # create mask where min distance < params.EQUAL_KEYPOINT_THRESHOLD
        mask: np.ndarray = mins > params.EQUAL_KEYPOINT_THRESHOLD

        new_C = new_keypoints[mask, :]
        new_F = new_C
        # camera pose is the same for each
        new_T = np.tile(current_camera_pose, (new_C.shape[0], 1))

        if print_stats:
            print(
                f"UPDATE LANDMARKS: {tracked_C.shape[0]}/{state.C.shape[1]} existing candidates updated."
                f"{state.C.shape[1]-status.sum()} rejected by KLT. {status.sum()-inlier_mask.sum()} rejected by RANSAC."
            )
            print(
                f"UPDATE LANDMARKS: {new_C.shape[0]}/{new_keypoints.shape[0]} candidates added new."
            )
        # plot_image_and_points(img_new, tracked_C, new_C, new_keypoints[mask==False, :])
    else:
        tracked_C = np.zeros((0, 2))
        tracked_F = np.zeros((0, 2))
        tracked_T = np.zeros((0, 16))
        # if not add all the candidates
        new_C = new_keypoints
        new_F = new_C
        # camera pose is the same for each
        new_T = np.tile(current_camera_pose, (new_C.shape[0], 1))
        if print_stats:
            print(
                f"UPDATE LANDMARKS: No previous candidates, all {new_C.shape[0]} candidates added."
            )

    # transpose to make them 2 X N
    new_C = np.vstack([tracked_C, new_C])
    new_F = np.vstack([tracked_F, new_F])
    new_T = np.vstack([tracked_T, new_T])
    state.update_candidates(new_C.T, new_F.T, new_T.T)
    return state


def triangulate_candidates(
    old_state: State,
    current_camera_pose: np.ndarray,
    K: np.ndarray,
    print_stats: bool = False,
) -> State:
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
                f"UPDATE LANDMARKS: Number of landmarks currently tracked is {old_state.P.shape[1]}/{params.NUM_LANDMARKS_GOAL}. Not adding more."
            )
        return old_state
    else:
        num_to_add: int = min(
            params.LIMIT_NEW_LANDMARKS, params.NUM_LANDMARKS_GOAL - current_landmarks
        )

    angles: np.ndarray = np.zeros((old_state.C.shape[1]))
    # do projections
    for i, (C, F, T) in enumerate(zip(old_state.C.T, old_state.F.T, old_state.T.T)):
        v_orig = unit_vector_to_pixel_in_world(K, T.reshape((4, 4)), F)
        v_curr = unit_vector_to_pixel_in_world(
            K, current_camera_pose.reshape((4, 4)), C
        )
        angles[i] = angle_between_units(v_orig, v_curr)

    # assumption: larger angle => better landmark to start tracking
    # take the num_to_add largest angles, set the rest to zero
    # This gets points close to the camera.
    small_indices = np.argsort(angles)[: (angles.shape[0] - num_to_add)]
    angles[small_indices] = 0

    # shuffled = np.random.permutation(angles.shape[0])
    # angles = angles[shuffled]
    # angles[: (angles.shape[0] - num_to_add)] = 0

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
    means = np.mean(old_state.X, axis=1)
    stds = np.std(old_state.X, axis=1)
    num_successfully_added = 0
    behind_camera = 0
    for i, (C, F, T) in enumerate(zip(tri_C.T, tri_F.T, tri_T.T)):
        new_X, mask = triangulate_points_wrapper(
            T.reshape(4, 4), current_camera_pose.reshape(4, 4), K, F, C
        )
        if not mask.all():
            behind_camera += 1
            continue
        # See if the landmark is a significant outlier from our existing 3D points.
        z_score = np.average(np.abs((new_X.T - means) / stds))
        if z_score > params.OUTLIER_3D_REJECTION_SIGMA:
            # if print_stats:
            #     print(
            #         f"UPDATE LANDMARKS: rejecting triangulated new keypoint \n"
            #         f"{new_X} \nbecause it is an outlier from the rest of state.X"
            #     )
            continue
        num_successfully_added += 1
        old_state.add_landmark(C.reshape(2, -1), new_X.reshape(3, -1))

    if print_stats:
        print(
            f"UPDATE LANDMARKS: Of {old_state.C.shape[1]} candidates, we attempted to add {tri_C.shape[1]} and "
            f"{num_successfully_added} were triangulated and added to state.(X/P). {behind_camera} were behind the camera."
        )
        print(
            f"UPDATE LANDMARKS: Number of landmarks now tracked is {old_state.P.shape[1]}/{params.NUM_LANDMARKS_GOAL}."
        )

    # TODO refine X estimate with non linear optimisation
    # move candidate to X,P in state
    old_state.update_candidates(new_C, new_F, new_T)
    return old_state


def unit_vector_to_pixel_in_world(
    K: np.ndarray, T: np.ndarray, pixel: np.ndarray
) -> np.ndarray:
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
    vector_camera_coordinates = direction_camera_coordinates / np.linalg.norm(
        direction_camera_coordinates
    )

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


def plot_image_and_points(image, points1, points2, points3):
    """
    Creates a figure, plots an image, and two arrays of 2D points with different colors.

    Parameters:
    - image: 2D or 3D array representing the image.
    - points1: Array of 2D points to be plotted with color 'red'.
    - points2: Array of 2D points to be plotted with color 'blue'.
    """
    fig, ax = plt.subplots()

    # Plot the image
    ax.imshow(image)  # Use 'gray' colormap for grayscale images

    if len(points1) > 0:
        ax.scatter(points1[:, 0], points1[:, 1], color="blue", label="Tracked points")

    for point in points1:
        circle = plt.Circle(point, radius=8, fill=False, color="blue")
        ax.add_patch(circle)

    if len(points2) > 0:
        ax.scatter(points2[:, 0], points2[:, 1], color="lime", label="Added new", s=4)
    if len(points3) > 0:
        ax.scatter(points3[:, 0], points3[:, 1], color="red", label="Rejected new", s=4)

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # Show the plot
    plt.show()

    print("done")
