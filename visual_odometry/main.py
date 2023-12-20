import cv2
import numpy as np
import matplotlib.pyplot as plt


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
from utils.visualisation import update_visualization


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


def main_loop():
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
    # Setup visualiser  #
    ###############################

    if not params.DO_PROFILING:
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

    ###############################
    # Continuous Visual Odometry  #
    ###############################

    current_state: State = initial_state
    prev_image: np.ndarray = None
    generator = image_generator(folder_path)

    camera_pose_history = np.zeros((3, len(os.listdir(folder_path))))
    global_point_cloud: set[tuple[float, float, float]] = set()
    prev_image = None
    R = None
    t = None
    for idx, filename, new_image, color_image in generator:
        if params.LIMIT_FRAME_COUNT and idx > params.FRAME_LIMIT:
            break
        if params.PRINT_STATS:
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
        R, t = estimating_current_pose(
            current_state, K, visualization=False, refine_with_DLT=params.REFINE_POSE
        )
        curr_pose: np.ndarray = create_homogeneous_matrix(R, t).flatten()
        # convert t to world frame for plotting
        t_W = -R.T @ t
        camera_pose_history[:, idx] = t_W.squeeze()
        num_landmarks = current_state.P.shape[1]
        current_state = update_landmarks(
            current_state,
            prev_image,
            new_image,
            curr_pose,
            K,
            print_stats=params.PRINT_STATS,
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


if __name__ == "__main__":
    if params.DO_PROFILING:
        profiler = cProfile.Profile()
        profiler.enable()

    main_loop()

    if params.DO_PROFILING:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("ncalls")
        # stats.strip_dirs()
        stats.dump_stats("full_run.stats")
        # stats.print_stats()
