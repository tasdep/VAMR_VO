import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import os
import sys

import params.params as params

from bootstrapping.initialization import initialize_pipeline
from continuous_vo.associate_keypoints_to_existing_landmarks import track_and_update
from continuous_vo.estimating_current_pose import estimating_current_pose
from continuous_vo.ransacLocalization_2 import ransacLocalization

from utils.state import State


def update_visualization(
    fig, ax1, ax2, ax3, current_image, state, R, t, camera_pose_history, R_ground, t_ground, camera_pose_history_ground, idx
):
    # Update Plot 1: Current image with keypoints
    ax1.clear()
    ax1.set_title("Current Image with Keypoints in Frame: "+str(idx))
    img_with_keypoints = current_image
    for p in state.P.T:
        p = p.astype(int)
        cv2.circle(
            img_with_keypoints,
            (p[1], p[0]),
            radius=3,
            color=(0, 255, 0),
            thickness=-1,
        )
    if state.C:
        for c in state.C.T:
            c = c.astype(int)
            cv2.circle(
                img_with_keypoints,
                (c[1], c[0]),
                radius=3,
                color=(255, 0, 0),
                thickness=-1,
            )
    ax1.imshow(img_with_keypoints)

    # Update Plot 2: 3D Point Cloud with Camera Pose
    ax2.clear()
    ax2.set_title("3D Point Cloud with Camera Pose")
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.set_zlabel("Z axis")
    print(state.X.shape)
    ax2.scatter(state.X[0, :], state.X[1, :], state.X[2, :])
    cam_length = 10
    for i in range(3):
        direction = R[:, i]
        ax2.quiver(*t.ravel(), *direction, length=cam_length, color=["r", "g", "b"][i])

    # Set fixed limits for the axes
    # You need to mess with this depending on the dataset.
    ax2.set_xlim([-100, 100])
    ax2.set_ylim([-100, 100])
    ax2.set_zlim([-100, 100])

    # Set a consistent orientation
    ax2.view_init(elev=-80, azim=90)

    # Add a marker for the origin
    ax2.scatter([0], [0], [0], color="k", marker="o")  # Black dot at the origin

    # Update Plot 3: 2D Camera Pose History
    ax3.clear()
    ax3.set_title("2D Top-Down Camera Pose History")
    ax3.set_xlabel("X axis")
    ax3.set_ylabel("Y axis")
    ax3.plot(camera_pose_history[0, :], camera_pose_history[1, :], label="Ours")
    #ax3.plot(camera_pose_history_ground[0, :], camera_pose_history_ground[1, :], label="Exercise")
    ax3.scatter(camera_pose_history[0, :], camera_pose_history[1, :], s=10, c='r', marker='o')
    ax3.legend()

    plt.draw()
    plt.pause(0.001)  # Necessary for the plot to update
    plt.waitforbuttonpress()  # Wait for keyboard input


def image_generator(folder_path):
    """
    Generator function to load images from a folder sequentially.

    Args:
    - folder_path (str): Path to the folder containing images.

    Yields:
    - Tuple[str, np.ndarray]: Image filename and corresponding image array.
    """
    idx = -1
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif")):
            idx += 1
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            yield idx, filename, image, color_image


if __name__ == "__main__":
    ###############################
    # Load data                   #
    ###############################
    base_path = Path(__file__).parent
    match params.DATASET:
        case params.Dataset.DATASET1:
            folder_path = (base_path / "shared_data/parking/images").resolve()
            calib_path = (base_path / "shared_data/parking/K.txt").resolve()
            K: np.ndarray = np.genfromtxt(
                calib_path, delimiter=",", dtype=float
            )  # calibration matrix[3x3]
            pass
        case params.Dataset.DATASET2:
            pass
        case params.Dataset.DATASET3:
            pass
        case params.Dataset.DATASET4:
            folder_path = (base_path / "local_data/test_data").resolve()
            calib_path = (base_path / "local_data/test_data/K.txt").resolve()
            K: np.ndarray = np.genfromtxt(
                calib_path, delimiter=" ", dtype=float
            )  # calibration matrix[3x3]

    ###############################
    # Bootstrap Initial Landmarks #
    ###############################
    if params.SKIP_BOOTSTRAP:
        if params.DATASET != params.Dataset.DATASET4:
            print(
                "Error: Must use Dataset 4 (local_data from ransac ex) for skipping initialization"
            )
            os._exit()
        keypoints = np.loadtxt(
            (base_path / "local_data/test_data/keypoints.txt").resolve()
        ).T
        p_W_landmarks = np.loadtxt(
            (base_path / "local_data/test_data/p_W_landmarks.txt").resolve()
        ).T
        initial_state = State()
        initial_state.update_landmarks(p_W_landmarks, keypoints)

    else:
        generator = image_generator(folder_path)
        images: list[np.ndarray] = []
        for _ in range(10):
            _, _, image, _ = next(generator)
            images.append(image)
        images: np.ndarray = np.array(images)

        initial_state: State
        initial_pose: np.ndarray
        initial_state, initial_pose = initialize_pipeline(
            images, K, visualise=False, print_stats=True
        )

    ###############################
    # Continuous Visual Odometry  #
    ###############################

    # Initialize visualization
    # Create a figure and subplots outside the function
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0], projection="3d")
    ax3 = fig.add_subplot(gs[1, 1])

    current_state: State = initial_state
    prev_image: np.ndarray = None
    camera_poses_history: list[np.ndarray] = []
    generator = image_generator(folder_path)

    camera_pose_history = np.zeros((3, len(os.listdir(folder_path))))
    camera_pose_history_ground = np.zeros((3, len(os.listdir(folder_path))))

    prev_image = None
    for idx, filename, new_image, color_image in generator:
        print(f"Analyzing image {idx}")
        if prev_image is None:
            prev_image = new_image
            continue
        track_and_update(
            current_state,
            prev_image,
            new_image,
            visualize=False,
            visualizer_img=color_image,
        )

        R, t = estimating_current_pose(current_state, K, visualization=False)
        R_ground, t_ground = ransacLocalization(current_state, K)

        camera_pose_history[:, idx] = t.squeeze()
        camera_pose_history_ground[:, idx] = t_ground.squeeze()

        update_visualization(
            fig,
            ax1,
            ax2,
            ax3,
            color_image,
            current_state,
            R,
            t,
            camera_pose_history[:, :idx],
            R_ground,
            t_ground,
            camera_pose_history_ground[:, :idx],
            idx,
        )

        prev_image = new_image

