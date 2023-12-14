import cv2
import numpy as np
from utils.state import State
from params import params as params


def track_and_update(
    state: State,
    prev_image: np.ndarray,
    new_image: np.ndarray,
    visualize: bool = True,
    visualizer_img: np.ndarray = None,
):
    """
    Tracks keypoints from the previous frame in the new frame using KLT tracker and updates the state.

    Parameters:
        state (State): The current state of the visual odometry system.
        new_image (np.ndarray): The new image frame in which keypoints need to be tracked.
        prev_image (np.ndarray): The previous image in which the state's keypoints are located.
        visualize (bool): Whether to show the tracked keypoints overlayed on an image.
        visualizer_img (np.ndarray): Image to overlay tracked keypoints on. Should be opened with
            cv2.IMREAD_COLOR for colored tracking traces.

    Returns:
        None: The function updates the state object in place.
    """

    # Ensure the state object has keypoints to track
    if state.P is None or state.P.shape[1] == 0:
        print("No keypoints to track in the state.")
        return

    # Convert keypoints from 2xN to Nx1x2 format for OpenCV
    prev_keypoints = state.P.T.reshape(-1, 1, 2)
    prev_keypoints = np.array(prev_keypoints, dtype=np.float32)

    # Parameters for KLT tracker
    lk_params = {
        "winSize": params.KLT_WINDOW_SIZE,
        "maxLevel": params.KLT_MAX_LEVEL,
        "criteria": params.KLT_CRITERIA,
    }

    # Track the keypoints in the new image
    # new_keypoints.shape = Nx1x2
    new_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_image, new_image, prev_keypoints, None, **lk_params
    )

    # Filter out the keypoints for which tracking was successful
    tracked_new = new_keypoints[status == 1]  # Shape: Nx2
    tracked_prev = prev_keypoints[status == 1]  # Shape: Nx2
    tracked_3D = state.X[:, status.flatten().astype(bool)]  # Shape: 3xN

    if visualize:
        # Squeeze the arrays to get rid of the added dimension.
        visualize_keypoint_movement_with_status(
            "Keypoint Movement with Status",
            visualizer_img,
            prev_keypoints.squeeze(),
            new_keypoints.squeeze(),
            status,
        )

    # Use RANSAC to estimate the transformation between prev_image and
    # new_image, and find one with the least outliers.
    # TODO: Check this is the intended way they want us to use RANSAC and that we are
    # allowed to use this opencv function.
    _, inlier_mask = cv2.findFundamentalMat(tracked_prev, tracked_new, cv2.FM_RANSAC)
    if not np.sum(inlier_mask):
        print("Error: No Inliers found")
        return
    # ravel is a broadcasting thing to make inlier_mask 1d so it can be used across each
    # row of the arrays it's indexing.
    inlier_new = tracked_new[inlier_mask.ravel() == 1]  # Shape Nx2
    inlier_prev = tracked_prev[inlier_mask.ravel() == 1]  # Shape Nx2
    inlier_3D = tracked_3D[:, inlier_mask.flatten().astype(bool)]  # Shape 3xN

    # Get rid of outliers in terms of distance moved in 2D
    # Note: I (kappi) just made this up. Feel free to question.
    euclidean_distances = np.linalg.norm(inlier_new - inlier_prev, axis=1)
    movement_outliers = (
        np.abs(euclidean_distances - euclidean_distances.mean())
        > params.TRACKING_OUTLIER_REJECTION_SIGMA * euclidean_distances.std()
    )

    
    inlier_new = inlier_new[~movement_outliers]  # Shape Nx2
    inlier_prev = inlier_prev[~movement_outliers]  # Shape Nx2
    inlier_3D = inlier_3D[:, ~movement_outliers]  # Shape 3xN

    if visualize:
        visualize_keypoint_movement_with_status(
            "After running ransac",
            visualizer_img,
            inlier_prev,
            inlier_new,
            np.ones((inlier_new.shape[0], 1)),
        )

    # Update keypoints in the state
    # Transform P back to 2xN format
    state.update_landmarks(inlier_3D, inlier_new.T)


def visualize_keypoint_movement_with_status(
    title: str,
    new_image: np.ndarray,
    prev_keypoints: np.ndarray,
    new_keypoints: np.ndarray,
    status: np.ndarray,
):
    """
    Visualizes the movement of keypoints on a single image with distinction between successfully and
    unsuccessfully tracked keypoints.

    Parameters:
        title (str): The title to show on the plot.
        new_image (np.ndarray): The new image frame where keypoints are to be visualized.
        prev_keypoints (np.ndarray): The keypoints in the previous image, in the shape returned by OpenCV: (Mx2)
        new_keypoints (np.ndarray): The keypoints in the new image, in the shape returned by OpenCV: (Mx2)
        status (np.ndarray): Array indicating whether each keypoint was successfully tracked, shape (Mx1)

    Returns:
        None: The function displays the visualization.
    """
    vis_image = new_image.copy()

    # Iterate through keypoints and status
    for p1, p2, s in zip(prev_keypoints, new_keypoints, status):
        # Convert to integer tuples
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))

        if s[0] == 1:  # Successfully tracked
            cv2.circle(
                vis_image, p1, 5, (0, 255, 0), -1
            )  # Green for previous keypoints
            cv2.circle(vis_image, p2, 5, (255, 0, 0), -1)  # Blue for new keypoints
            cv2.line(
                vis_image, p1, p2, (255, 255, 0), 2
            )  # Cyan line to indicate movement
        else:  # Not successfully tracked
            cv2.circle(
                vis_image, p1, 5, (0, 0, 255), -1
            )  # Red for previous keypoints not tracked

    # Display the image
    cv2.imshow(title, vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
