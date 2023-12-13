from bootstrapping.initialization import initialize_pipeline
from utils.state import State

import numpy as np
import os
import cv2

def read_images_into_array(images_directory: str, n: int) -> np.ndarray:
    # Initialize an empty list to store the grayscale images
    grayscale_images: list[np.ndarray] = []

    # Loop through the images in the directory
    for i in range(n):  # Assuming your images are named from 0 to 9
        # Construct the path to each image
        image_path: str = os.path.join(images_directory, f"img_{i:05d}.png")

        # Read the image in grayscale
        img_gray: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is not None:
            # Append the grayscale image to the list
            grayscale_images.append(img_gray)
        else:
            print(f"Error reading image {i}")

    # Convert the list of grayscale images to a numpy array
    numpy_array: np.ndarray = np.array(grayscale_images)

    return numpy_array

def shared_data_test():
        # Specify the path to the directory containing your images
    images_directory: str = "..\shared_data\parking\images"


    # Read images into a numpy array
    images: np.ndarray = read_images_into_array(images_directory, 10)
    K: np.ndarray = np.genfromtxt("..\shared_data\parking\K.txt", delimiter=',', dtype=float)  # calibration matrix[3x3]
    
    # cv2.imshow("temp", images[0,...])
    initial_state: State = initialize_pipeline(images, K, print_stats=True, visualise=True)

def ex6_triangulation_test():
    # load exercise images
    img_1 = np.array(cv2.imread('..\local_data\ex6_data\\0001.jpg', cv2.IMREAD_GRAYSCALE))
    img_2 = np.array(cv2.imread('..\local_data\ex6_data\\0002.jpg', cv2.IMREAD_GRAYSCALE))

    K = np.array([  [1379.74,   0,          760.35],
                    [    0,     1382.08,    503.41],
                    [    0,     0,          1 ]] )
    # load exercise 2D keypoints
    p1 = np.loadtxt('..\local_data\ex6_data\matches0001.txt')
    p2 = np.loadtxt('..\local_data\ex6_data\matches0002.txt')

    images = np.array([img_1, img_2, img_2, img_2])
    print(f"{images.shape=}")
    
    # initial_state: State = initialize_pipeline(images, K, print_stats=True, visualise=True,prematached_keypoints=[p1.T,p2.T])
    initial_state: State = initialize_pipeline(images, K, print_stats=True, visualise=True)

    pass
    
if __name__ == "__main__":
    # ex6_triangulation_test()
    shared_data_test()
    print("Done")
