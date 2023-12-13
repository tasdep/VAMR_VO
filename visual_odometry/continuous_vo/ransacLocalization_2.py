import cv2
import numpy as np

from code_previous_exercises.estimate_pose_dlt import estimatePoseDLT
from code_previous_exercises.projectPoints import projectPoints


def ransacLocalization(state, K):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.
    """
    pass
    use_p3p = True
    tweaked_for_more = True
    adaptive = True   # whether or not to use ransac adaptively

    if use_p3p:
        num_iterations = 1000 if tweaked_for_more else 200
        pixel_tolerance = 10
        k = 3
    else:
        num_iterations = 2000
        pixel_tolerance = 10
        k = 6

    if adaptive:
        num_iterations = float('inf')

    # Initialize RANSAC

    matched_query_keypoints = state._P
    corresponding_landmarks = state._X.T

    best_inlier_mask = np.zeros(matched_query_keypoints.shape[1])
    # (row, col) to (u, v)
    matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)
    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    i = 0
    while num_iterations > i:
        # Model from k samples (DLT or P3P)
        indices = np.random.permutation(corresponding_landmarks.shape[0])[:k]
        landmark_sample = corresponding_landmarks[indices, :]
        keypoint_sample = matched_query_keypoints[:, indices]

        if use_p3p:
            success, rotation_vectors, translation_vectors = cv2.solveP3P(landmark_sample, keypoint_sample.T, K,
                                                                        None, flags=cv2.SOLVEPNP_P3P)
            t_C_W_guess = []
            R_C_W_guess = []
            for rotation_vector in rotation_vectors:
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                for translation_vector in translation_vectors:
                    R_C_W_guess.append(rotation_matrix)
                    t_C_W_guess.append(translation_vector)

        else:
            M_C_W_guess = estimatePoseDLT(keypoint_sample.T, landmark_sample, K)
            R_C_W_guess = M_C_W_guess[:, :3]
            t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        if not use_p3p:
            C_landmarks = np.matmul(R_C_W_guess, corresponding_landmarks[:, :, None]).squeeze(-1) + t_C_W_guess[None, :]
            projected_points = projectPoints(C_landmarks, K)
            difference = matched_query_keypoints - projected_points.T
            errors = (difference**2).sum(0)
            is_inlier = errors < pixel_tolerance**2

        else:
            # If we use p3p, also consider inliers for the 4 solutions.
            is_inlier = np.zeros(corresponding_landmarks.shape[0])
            for alt_idx in range(len(R_C_W_guess)):
                C_landmarks = np.matmul(R_C_W_guess[alt_idx], corresponding_landmarks[:, :, None]).squeeze(-1) + \
                              t_C_W_guess[alt_idx][None, :].squeeze(-1)
                projected_points = projectPoints(C_landmarks, K)
                difference = matched_query_keypoints - projected_points.T
                errors = (difference ** 2).sum(0)
                alternative_is_inlier = errors < pixel_tolerance ** 2
                if alternative_is_inlier.sum() > is_inlier.sum():
                    is_inlier = alternative_is_inlier

        min_inlier_count = 30 if tweaked_for_more else 6

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        if adaptive:
            # estimate of the outlier ratio
            outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]
            # formula to compute number of iterations from estimated outlier ratio
            confidence = 0.95
            upper_bound_on_outlier_ratio = 0.90
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            num_iterations = np.log(1-confidence)/np.log(1-(1-outlier_ratio)**k)
            # cap the number of iterations at 15000
            num_iterations = min(15000, num_iterations)

        num_iteration_history.append(num_iterations)
        max_num_inliers_history.append(max_num_inliers)

        i += 1
    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W  = estimatePoseDLT(matched_query_keypoints[:, best_inlier_mask].T, corresponding_landmarks[best_inlier_mask, :], K)
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

        if adaptive:
            print("    Adaptive RANSAC: Needed {} iteration to converge.".format(i - 1))
            print("    Adaptive RANSAC: Estimated Ouliers: {} %".format(100 * outlier_ratio))

    return R_C_W.T, t_C_W
