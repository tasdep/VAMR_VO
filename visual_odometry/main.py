import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from pathlib import Path
import os
import cProfile
import pstats

import params.params as params

from bootstrapping.initialization import initialize_pipeline
from continuous_vo.associate_keypoints_to_existing_landmarks import track_and_update
from continuous_vo.estimating_current_pose import estimating_current_pose
from continuous_vo.update_landmarks import update_landmarks

from utils.state import State
from utils.utils import create_homogeneous_matrix
from utils.visualisation import drawCamera


def update_visualization(
    fig,
    ax1,
    ax2,
    ax3,
    current_image,
    state,
    previous_keypoint_locations,
    current_locations,
    global_point_cloud,
    R,
    t,
    camera_pose_history,
    num_added_landmarks,
    idx,
):
    """The main visualization function for the pipeline."""
    # Update Plot 1: Current image with keypoints
    ax1.clear()
    ax1.set_title(f"Image {idx}. With {num_added_landmarks} new landmarks")
    num_old_landmarks = state.P.shape[1] - num_added_landmarks

    idx = 0
    for p1, p2 in zip(previous_keypoint_locations, current_locations):
        color = (0, 255, 0)
        idx += 1
        # Convert to integer tuples
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))

        cv2.circle(current_image, p1, 5, (0, 0, 255), -1)  # Blue for previous keypoints
        cv2.circle(current_image, p2, 5, color, -1)  # Green for new keypoints
        cv2.line(current_image, p1, p2, color, 2)  # line to indicate movement

    for idx, p in enumerate(state.P.T):  # state.P.T[num_old_landmarks:, :]):
        color = (255, 132, 0) if idx < num_old_landmarks else (255, 0, 0)
        p = p.astype(int)
        cv2.circle(
            current_image,
            (p[0], p[1]),
            radius=5,
            color=color,
            thickness=-1,
        )

    if state.C is not None:
        for c in state.C.T:
            c = c.astype(int)
            cv2.circle(
                current_image,
                (c[0], c[1]),
                radius=2,
                color=(50, 168, 155),
                thickness=-1,
            )
    # Create proxy artists for the legend
    green_circle = Line2D(
        [0], [0], linestyle="none", marker="o", markersize=6, markerfacecolor=(0, 1, 0)
    )
    red_circle = Line2D(
        [0], [0], linestyle="none", marker="o", markersize=6, markerfacecolor=(1, 0, 0)
    )
    cyan_circle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=3,
        markerfacecolor=(50 / 255, 168 / 255, 155 / 255),
    )
    orange_cricle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=6,
        markerfacecolor=(237 / 255, 137 / 255, 7 / 255),  # Converted to range 0-1
    )

    # Add the legend to the plot
    ax1.legend(
        [orange_cricle, green_circle, red_circle, cyan_circle],
        [
            "Landmarks tracked in front of cam.",
            "Landmarks tracked behind cam, remvoed",
            "Promoted Candidates",
            "Candidates",
        ],
        numpoints=1,
        fontsize="small",
    )

    ax1.imshow(current_image)

    # Update Plot 2: 3D Point Cloud with Camera Pose
    ax2.clear()
    ax2.set_title("3D Point Cloud with Camera Pose")
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.set_zlabel("Z axis")

    if params.GLOBAL_POINT_CLOUD:
        global_x = [point[0] for point in global_point_cloud]
        global_y = [point[1] for point in global_point_cloud]
        global_z = [point[2] for point in global_point_cloud]
        ax2.scatter(global_x, global_y, global_z, s=2)
        ax2.set_box_aspect((np.ptp(global_x), np.ptp(global_y), np.ptp(global_z)))
    else:
        ax2.scatter(state.X[0, :], state.X[1, :], state.X[2, :])
        ax2.set_box_aspect(
            (np.ptp(state.X[0, :]), np.ptp(state.X[1, :]), np.ptp(state.X[2, :]))
        )
    drawCamera(
        ax2,
        (-R.T @ t).ravel(),
        R.T,
        length_scale=20,
        head_size=10,
        equal_axis=False,
        set_ax_limits=False,
    )

    # Add a marker for the origin
    ax2.scatter([0], [0], [0], color="k", marker="o")  # Black dot at the origin

    ax2.plot(
        camera_pose_history[0, :],
        camera_pose_history[1, :],
        camera_pose_history[2, :],
        color="red",
    )  # Line plot

    # Update Plot 3: 2D Camera Pose History
    ax3.clear()
    ax3.set_title("2D Top-Down Camera Pose History")
    ax3.set_xlabel("X axis")
    ax3.set_ylabel("Z axis")
    ax3.plot(camera_pose_history[0, :], camera_pose_history[2, :])
    ax3.set_aspect("equal")
    ax3.autoscale(enable=True, axis="both")

    plt.draw()
    plt.pause(0.001)  # Necessary for the plot to update
    if params.WAIT_ARROW:
        fig.canvas.start_event_loop(timeout=-1)
    else:
        # do nothing since we are going to compute the next frame
        pass


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
            if idx < params.START_IMG_IDX:
                continue
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
            folder_path = (base_path / "../shared_data/parking/images").resolve()
            calib_path = (base_path / "../shared_data/parking/K.txt").resolve()
            K: np.ndarray = np.genfromtxt(
                calib_path, delimiter=",", dtype=float
            )  # calibration matrix[3x3]
            pass
        case params.Dataset.DATASET2:
            folder_path = (base_path / "../local_data/kitti/05/image_0").resolve()
            calib_path = (base_path / "../local_data/kitti/05/calib.txt").resolve()
            all_calib: np.ndarray = np.genfromtxt(
                calib_path, delimiter=" ", dtype=float
            )
            K: np.ndarray = all_calib[0, 1:].reshape([3, 4])[
                :, :-1
            ]  # calibration matrix[3x3]
            pass
        case params.Dataset.DATASET3:
            pass
        case params.Dataset.DATASET4:
            folder_path = (base_path / "../local_data/test_data").resolve()
            calib_path = (base_path / "../local_data/test_data/K.txt").resolve()
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
            (base_path / "../local_data/test_data/keypoints.txt").resolve()
        ).T
        keypoints = keypoints[[1, 0], :]
        p_W_landmarks = np.loadtxt(
            (base_path / "../local_data/test_data/p_W_landmarks.txt").resolve()
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

    if params.DO_PROFILING:
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        # Initialize visualization
        # Create a figure and subplots outside the function
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(figsize=(10, 8))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0], projection="3d")
        ax3 = fig.add_subplot(gs[1, 1])
        # Set a consistent orientation
        ax2.view_init(elev=-70, azim=-90)
        # setup plot event
        def on_key(event):
            if event.key == "right" and params.WAIT_ARROW:
                print("Right arrow key pressed.")
                fig.canvas.stop_event_loop()
            elif event.key == "escape":
                print("Esc key pressed.")
                plt.close("all")
                exit(0)

            plt.connect("key_press_event", on_key)

    current_state: State = initial_state
    prev_image: np.ndarray = None
    camera_poses_history: list[np.ndarray] = []
    generator = image_generator(folder_path)

    camera_pose_history = np.zeros((3, len(os.listdir(folder_path))))
    global_point_cloud: set[tuple[float, float, float]] = set()
    prev_image = None
    R = None
    t = None
    for idx, filename, new_image, color_image in generator:
        if params.LIMIT_FRAME_COUNT and idx > params.FRAME_LIMIT:
            break
        print(f"MAIN LOOP: Analyzing image {idx}")
        if prev_image is None:
            prev_image = new_image
            continue
        previous_keypoint_locations, current_locations = track_and_update(
            current_state,
            prev_image,
            new_image,
            visualize=False,
            visualizer_img=color_image,
        )
        # fig_cp, ax_cp = plt.subplots()
        # figure = fig_cp
        # image = color_image
        R, t = estimating_current_pose(current_state, K, visualization=False)
        curr_pose: np.ndarray = create_homogeneous_matrix(R, t).flatten()
        # convert t to world frame for plotting
        t_W = -R.T @ t
        camera_pose_history[:, idx] = t_W.squeeze()
        num_landmarks = current_state.P.shape[1]
        current_state = update_landmarks(
            current_state, prev_image, new_image, curr_pose, K, print_stats=True
        )
        added_landmarks = current_state.P.shape[1] - num_landmarks
        if not params.DO_PROFILING:
            # Update the global point cloud
            if params.GLOBAL_POINT_CLOUD:
                for point in current_state.X.T:
                    global_point_cloud.add((point[0], point[1], point[2]))
            update_visualization(
                fig,
                ax1,
                ax2,
                ax3,
                color_image,
                current_state,
                previous_keypoint_locations,
                current_locations,
                global_point_cloud,
                R,
                t,
                camera_pose_history[:, : min(camera_pose_history.shape[1], idx + 1)],
                added_landmarks,
                idx,
            )

        prev_image = new_image

    if params.DO_PROFILING:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("ncalls")
        # stats.strip_dirs()
        stats.dump_stats("full_run.stats")
        # stats.print_stats()
