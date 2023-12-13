import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    
    # Convert 2D to normalized coordinates
    p_norm = (np.linalg.inv(K) @ np.c_[p, np.ones((p.shape[0], 1))].T).T

    # Build measurement matrix Q
    num_corners = p_norm.shape[0]
    Q = np.zeros((2*num_corners, 12))

    for i in range(num_corners):
        u = p_norm[i, 0]
        v = p_norm[i, 1]

        Q[2*i, 0:3] = P[i,:]
        Q[2*i, 3] = 1
        Q[2*i, 8:11] = -u * P[i,:]
        Q[2*i, 11] = -u
        
        Q[2*i+1, 4:7] = P[i,:]
        Q[2*i+1, 7] = 1
        Q[2*i+1, 8:11] = -v * P[i,:]
        Q[2*i+1, 11] = -v

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    u, s, v = np.linalg.svd(Q, full_matrices=True)
    M_tilde = np.reshape(v.T[:,-1], (3,4));
    
    # Extract [R | t] with the correct scale
    if (np.linalg.det(M_tilde[:, :3]) < 0):
        M_tilde *= -1

    R = M_tilde[:, :3]

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    u, s, v = np.linalg.svd(R);
    R_tilde = u @ v;

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_tilde, 'fro')/np.linalg.norm(R, 'fro');

    t_tilde = alpha * M_tilde[:,3]

    # Build M_tilde with the corrected rotation and scale
    M_tilde = np.c_[R_tilde, t_tilde];
    
    return M_tilde
