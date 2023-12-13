import cv2
import numpy as np
from utils.state import State
from params import params as params

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def estimating_current_pose(
        state: State,
        K: np.ndarray,
        PnP_solver: str = 'P3P',
        refine_with_DLT: bool = False,
        visualization: bool = True,
        figure: Figure = None,
        image: np.ndarray = None
):
    """
    Estimates the current camera pose (localization) and inliers using RANSAC. 
    For pose estimation P3P and DLT can be used, but P3P is prefered.
    
    Parameters:
        state (State): Current state of the VO system.
        K (np.ndarray): Camera matrix. Dimension: 3 x 3
        PnP_solver (str): Can be either 'P3P' or 'DLT' to solve the PnP localization problem.
        refine_with_DLT (bool): Refine the rotation and translation with the determined inliers by RANSAC.
        visualization (bool): Can be used the visualize the camera pose.
        figure (Figure): 3D landmarks and camera pose visualization.
        image (np.ndarray): For inlier visualization.

    Returns:
        R (np.ndarray): 3x3 rotation matrix of the current frame.
        t (np.ndarray): 3x1 translation vector of the current frame.
        In the homogeneous pose matrix (4x4) configuration: 
        [R , t]
        [0 , 1]
    """

    # Valid methods for solving the PnP localization problem
    valid_methods = ['P3P', 'DLT']

    # Check if the provided method is valid
    if PnP_solver not in valid_methods:
        raise ValueError(f"Invalid method. Please choose from: {', '.join(valid_methods)}")

    # Initialize the rotation and translation
    R = np.zeros((3,3)) #dim 3x3
    t = np.zeros((3,1)) #dim 3x1

    # Get the 2D keypoints from the state
    keypoints = state._P.T #dim: 2xN

    # Reshape the keypoints from 2xN to Nx1x2 for cv2
    keypoints = np.reshape(keypoints, (-1, 1, 2)) #dim: Nx1x2

    # Get the corresponding 3D landmarks from the state
    landmarks = state._X.T #dim: 3xN

    # Reshape the landmarks from 3xN to Nx1x3 for cv2
    landmarks = np.reshape(landmarks, (-1, 1, 3)) #dim: Nx1x3

    # Initilaize RANSAC parameters:
    params_RANSAC = {
        'distCoeffs': None, # Distortion coefficients of the camera
        'iterationsCount': params.POSE_RANSAC_ITERATION, # Number of iterations for RANSAC
        'reprojectionError': params.POSE_RANSAC_REPROJECTION_ERROR, # Inlier threshold value used by the RANSAC procedure
        'confidence': params.POSE_RANSAC_CONFIDENCE # The probability that the algorithm produces a useful result
    }

    if PnP_solver == 'DLT':
        params_DLT = {
            'rvec': None, # Initial guess for the rotation
            'tvec': None, # Initial guess for the translation
            'useExtrinsicGuess': False, # Use initial guesses in the DLT algorithm
        }
        params_RANSAC.update(params_DLT)

    # decide the PnP solver
    PnP_flag = 0
    
    if PnP_solver == 'P3P':
        PnP_flag = cv2.SOLVEPNP_P3P
    elif PnP_solver == 'DLT':
        PnP_flag = cv2.SOLVEPNP_ITERATIVE
    else:
        raise ValueError(f"Invalid method. Please choose from: {', '.join(valid_methods)}")

    # cv2.SOLVEPNP_P3P or cv2.SOLVEPNP_ITERATIVE 
    succes, rotation_vector, translation_vector, inliers_index = cv2.solvePnPRansac(landmarks, keypoints, K, **params_RANSAC, flags=PnP_flag)

    # Print the number of inliers found:
    if succes == True:
        # If the RANSAC was successful append the inliers.
        number_of_points = landmarks.shape[0]
        number_of_inliers = len(inliers_index)

        # Find the inliers in the keypoints and the landmarks
        inlier_landmarks = np.zeros((number_of_inliers, landmarks.shape[1], landmarks.shape[2])) #dim: num_of_inliers x1x3
        inlier_keypoints = np.zeros((number_of_inliers, keypoints.shape[1], keypoints.shape[2])) #dim: num_of_inliers x1x2

        # Get out the inliers:
        for num_inlier, inliers in zip(range(number_of_inliers), inliers_index):
            inlier_landmarks[num_inlier, :, :] = landmarks[inliers, :, :]
            inlier_keypoints[num_inlier, :, :] = keypoints[inliers, :, :]

        # Create all of the indicies:
        full_indicies = np.arange(number_of_points).reshape(-1, 1)

        # Create the outlier indicies:
        outliers_index = np.setdiff1d(full_indicies, inliers_index).reshape(-1, 1)
        number_of_outliers = len(outliers_index)

        # Get the outliers:
        outlier_keypoints = np.zeros((number_of_outliers, keypoints.shape[1], keypoints.shape[2])) #dim: (number_of_points-num_of_inliers) x1x2

        for num_outlier, outliers in zip(range(number_of_outliers), outliers_index):
            outlier_keypoints[num_outlier, :, :] = keypoints[outliers, :, :]

        print('The RANSAC algorithm was succesful. Number of inliers found: ' + str(number_of_inliers) + ' ,out of: ' + str(number_of_points)+ '.')
    else:
        print('The RANSAC algorithm did not converged.')

    # Convert the rotation vector to a matrix
    R = cv2.Rodrigues(rotation_vector)[0] #dim: 3x3
    t = translation_vector

    # Refine the translation and rotation with the determined inliers.
    if refine_with_DLT == True:
        print('Refine the solution with the DLT algorithm.')

        #TODO: use useExtrinsicGuess with the RANSAC rotation and
        succes_DLT, rotation_vector_DLT, translation_vector_DLT = cv2.solvePnP(inlier_landmarks, inlier_keypoints, K, None, flags=cv2.SOLVEPNP_ITERATIVE)

        if succes_DLT == True:
            # Convert the rotation vector to a matrix
            R = cv2.Rodrigues(rotation_vector_DLT)[0] #dim: 3x3
            t = translation_vector_DLT #dim: 3x1

        else:
            print('The DLT refinement did not work. Use the original solution.')
            # Convert the rotation vector to a matrix
            R = cv2.Rodrigues(rotation_vector)[0] #dim: 3x3
            t = translation_vector #dim: 3x1

    else: 
        # Convert the rotation vector to a matrix
        R = cv2.Rodrigues(rotation_vector)[0] #dim: 3x3
        t = translation_vector #dim: 3x1

    
    # Visualize the results
    if visualization == True:
        visualize_pose(figure, image, R, t, landmarks, inlier_keypoints.T, outlier_keypoints.T)
    
    return R, t

def visualize_pose(
        figure: Figure, 
        image: np.ndarray, 
        R: np.ndarray, 
        t: np.ndarray, 
        landmarks: np.ndarray, 
        keypoints_inliers: np.ndarray,
        keypoints_outliers: np.ndarray
        ):
    """
    Visualization of the camera pose and for the inliers.

    Parameters:
        figure (Figure): 3D landmarks and camera pose visualization.
        image (np.ndarray): For inlier visualization.
        R (np.ndarray): Rotation matrix of the current frame. Dimension: 3x3
        t (np.ndarray): Translation vector of the current frame. Dimension: 3x1
        landmarks (np.ndarray): 3D landmarks. Dimension: Nx1x3
        keypoints_inliers: (np.ndarray): 2D keypoints of inliers after RANSAC. Dimension: Nx1x2
        keypoints_outliers: (np.ndarray): 2D keypoints of outliers after RANSAC. Dimension: Nx1x2

    Returns:
        None
    """
    
    # Create a figure and axis
    fig = figure
    fig.clear()

    # Plotting the data
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(image, cmap='gray')
    # # Plot the outliers
    # ax.plot(keypoints_outliers[:, :, 0], keypoints_outliers[:, :, 1], 'rx', linewidth=2)
    
    # # Plot the inliers if there are any.
    # if keypoints_inliers.size > 0:
    #      ax.plot(keypoints_inliers[:, :, 0], keypoints_inliers[:, :, 1], 'og', linewidth=0.5)

    # ax.set_title('Inlier and outlier matches')
    # ax.axis('off')

    camera_position = -np.matmul(R, np.squeeze(t))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(landmarks[:, :, 0], landmarks[:, :, 1], color='blue', marker='.',  s=1)
    ax1.scatter(camera_position[0], camera_position[1], color='red', marker='X', s=5)
    ax1.plot(camera_position[0], camera_position[1], color='green')
    ax1.set_xlim(-60,60)
    ax1.set_ylim(-60,60)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title("2D view")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(landmarks[:, :, 0], landmarks[:, :, 1], landmarks[:, :, 2], s=1)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    ax2.set_xlim3d(-60, 50)
    ax2.set_ylim3d(-60, 50)
    ax2.set_zlim3d(-60, 50)
    ax2.set_title("3D view")

    #drawCamera(ax2, camera_position, R, length_scale=10, head_size=10, equal_axis=True, set_ax_limits=True)
    cam_length = 10
    for i in range(3):
        direction = R[:, i]
        ax2.quiver(*t.ravel(), *direction, length=cam_length, color=["r", "g", "b"][i])

    plt.pause(1)
