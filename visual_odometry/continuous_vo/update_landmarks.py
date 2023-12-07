from utils.state import State
from utils.image_processing import run_harris_detector, describe_keypoints
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import params.params as params

def get_candidate_keypoints(P: np.ndarray, img_1: np.ndarray, visualise: bool = False, print_stats: bool = False) -> (np.ndarray,np.ndarray):
    #get the new keypoints with harris detector
    keypoints_new = run_harris_detector(img_1,visualise, print_stats)
    new_C = np.array(np.ndarray(keypoints_new.shape))
    new_descriptors = np.array(np.ndarray(keypoints_new.shape)) #descriptor jas same shape as detector ????
    #compare each keypoint in P with the new kepoints 
    # and only keep those that don't have a match in P
  
    for keypoint_new in keypoints_new:  
        flag = False
        for keypoint_old in P:
            #check similarity using ssd with a predefined threshold
            ssd = np.sum((keypoint_old-keypoint_new)**2)
            if(ssd <= params.KEYPOINT_THRESHOLD):
                flag = True
        if flag is False:
            np.append(new_C,keypoint_new)
            #save the descriptor of the new candidate keypoint
            new_descriptor = describe_keypoints(img_1, keypoints_new, params.DESC_PATCH_RAD)
            np.append(new_descriptors,new_descriptor)
            
    return new_C, new_descriptors


def get_updated_keypoints(state:State,current_camera_pose:np.ndarray, img_prev: np.ndarray,img_new: np.ndarray, new_C: np.ndarray, new_descriptors: np.ndarray) -> State:
    new_F = np.ndarray(new_C.shape)
    new_T = np.ndarray(new_C.shape)
    old_descriptors = describe_keypoints(img_prev, state.C, params.DESC_PATCH_RAD)
    
    for (indx_new, keypoint_new) in enumerate(new_C):  
        flag = False
        min_ssd = params.KEYPOINT_THRESHOLD +1 
        min_indx_old = -1
        for (indx_old, keypoint_old) in enumerate(state.C):
            #check similarity using ssd with a predefined threshold
            ssd = np.sum((old_descriptors[indx_old]-new_descriptors[indx_new])**2)  
            if(ssd <= params.KEYPOINT_THRESHOLD and ssd < min_ssd):
                min_ssd = ssd 
                min_indx_old = indx_old
                flag = True
        #No match
        if flag is False:
            np.append(new_F,keypoint_new)
            np.append(new_T,current_camera_pose)
        if flag is True:
            np.append(new_F,state.F[min_indx_old])
            np.append(new_T,state.T[min_indx_old])


            #check angle 
    return state.update_candidates(new_C,new_F,new_T)
