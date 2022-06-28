import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple


def _get_image_points(image, im_name: str, number_of_points: int, s: str) -> None:
    plt.figure()
    plt.imshow(image)
    plt.title(f'Choose {number_of_points} points (for set {s}) from the below image that you wish to transform')

    # Recall number_of_points = N
    image_points: List[tuple] = plt.ginput(n=number_of_points, timeout=0)  # Shape: (1 x 2N)
    plt.close()
    image_points: np.ndarray = np.array(image_points)  # Shape: (N x 2)

    # Reversing the coordinates (for in plt.ginput() function - x is the horizontal axis, and y is the vertical)
    # image_points = image_points[:, ::-1]  # Now the x coordinate in image_points fits an x coordinate in an image
    image_points = np.round(image_points)
    # ones_array = np.ones(number_of_points)  # Shape: (1 x N)

    # Adding a new dimension to each coordinate
    # image_points = np.column_stack([image_points, ones_array])  # Shape: (N x 3)

    np.save(im_name, image_points)


def get_image_pts(image_l, image_r, im_name_l: str, im_name_r: str, number_of_points: int = 10, s: str = 's1') -> None:
    _get_image_points(image_l, im_name_l, number_of_points, s)
    _get_image_points(image_r, im_name_r, number_of_points, s)


def load_pts(im_name_l: str, im_name_r: str) -> Tuple[np.ndarray, np.ndarray]:
    im_pts_l = np.load(im_name_l + ".npy")
    im_pts_r = np.load(im_name_r + ".npy")

    return im_pts_l, im_pts_r
