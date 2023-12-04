import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from mpl_toolkits.mplot3d import Axes3D
import os

import params.params as params

from bootstrapping.initialization import initialize_pipeline
from continuous_vo.associate_keypoints_to_existing_landmarks import track_and_update

from utils.state import State


class Dataset(Enum):
    DATASET1 = "Shared dataset"
    DATASET2 = "Kitti"
    DATASET3 = "Malaga"


dataset = Dataset.DATASET1


# Example usage:
# visualize_odometry(current_frame, keypoints_2d, points_3d, camera_pose, camera_poses_history)
def visualize_odometry(current_frame, keypoints_2d, points_3d, camera_pose, camera_poses_history):
    # Display the current frame with keypoints
    for point in keypoints_2d.T:
        cv2.circle(current_frame, (int(point[1]), int(point[0])), 5, (0, 255, 0), -1)

    cv2.imshow("Current Frame with Keypoints", current_frame)
    cv2.waitKey(1)  # Adjust the delay as needed, 0 for a keypress

    # Prepare for 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 3D points
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c="blue")

    # Plot current camera position
    cam_position = camera_pose[:3, 3]
    ax.scatter(cam_position[0], cam_position[1], cam_position[2], c="red", marker="o")

    # Plot camera poses over time
    ax.plot(camera_poses_history[0, :], camera_poses_history[1, :], camera_poses_history[2, :], color="green")

    # Setting labels for axes
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Show the plot
    plt.show()


def image_generator(folder_path):
    """
    Generator function to load images from a folder sequentially.

    Args:
    - folder_path (str): Path to the folder containing images.

    Yields:
    - Tuple[str, np.ndarray]: Image filename and corresponding image array.
    """
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            yield filename, image


if __name__ == "__main__":
    ###############################
    # Load data                   #
    ###############################
    match dataset:
        case Dataset.DATASET1:
            folder_path = "..\shared_data\parking\images"
            K: np.ndarray = np.genfromtxt("..\shared_data\parking\K.txt", delimiter=",", dtype=float)  # calibration matrix[3x3]
            pass
        case Dataset.DATASET2:
            pass
        case Dataset.DATASET3:
            pass

    ###############################
    # Bootstrap Initial Landmarks #
    ###############################
    generator = image_generator(folder_path)
    images: list[np.ndarray] = []
    for _ in range(10):
        _, image = next(generator)
        images.append(image)
    images: np.ndarray = np.array(images)

    initial_state: State = initialize_pipeline(images, K, visualise=False, print_stats=True)

    visualize_odometry(images[params.BOOTSRAP_FRAMES[1]], initial_state.P, initial_state.X, np.zeros((4, 4)), np.zeros((4, 4)))
    ###############################
    # Continuous Visual Odometry  #
    ###############################

    current_state: State = initial_state
    prev_image: np.ndarray = None
    camera_poses_history: list[np.ndarray] = []
    # loop through images
    # Example usage:
    generator = image_generator(folder_path)

    _, prev_image = next(generator)

    for filename, new_image in generator:
        # Process the image as needed
        if False:
            cv2.imshow("Image", image)
            cv2.waitKey(100)
        track_and_update(current_state, prev_image, new_image, True, new_image)
        # visualize_odometry()
        # run pipeline
        # visualise current image, actual current keypoints
        # 3D point cloud + current camera pose
        # track camera pose over time and plot in X,Y plane
