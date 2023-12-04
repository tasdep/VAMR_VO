from bootstrapping.initialization import initialize_pipeline
from utils.state import State

import numpy as np
import os
import cv2

def read_images_into_array(images_directory: str) -> np.ndarray:
    # Initialize an empty list to store the grayscale images
    grayscale_images: list[np.ndarray] = []

    # Loop through the images in the directory
    for i in range(10):  # Assuming your images are named from 0 to 9
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

if __name__ == "__main__":
    # Specify the path to the directory containing your images
    images_directory: str = "..\shared_data\parking\images"


    # Read images into a numpy array
    images: np.ndarray = read_images_into_array(images_directory)
    K: np.ndarray = np.genfromtxt("..\shared_data\parking\K.txt", delimiter=',', dtype=float)  # calibration matrix[3x3]
    
    # cv2.imshow("temp", images[0,...])
    initial_state: State = initialize_pipeline(images, K, print_stats=True, visualise=True)
