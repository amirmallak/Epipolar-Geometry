import os
import cv2
# import config
import utilities
import numpy as np
# import terminal_colors as tc
import matplotlib.pyplot as plt

from .epipolar_lines_functions import epipolar_distance, draw_lines


def calculate_epipolar_lines():
    # Font configurations
    # reset_all_style = tc.Style.RESET_ALL
    # fore_red = tc.Fg.RED
    # fore_green = tc.Fg.GREEN

    # data_path = config.epipolar_data_path
    data_path = os.path.join(r'.', 'images')
    images = [['hall_left.jpg', 'hall_right.jpg'],
              ['building_left.jpg', 'building_right.jpg']]
    set_points_names = ['s1', 's2']

    for im in images:
        for s in set_points_names:
            im_name_l, im_name_r = im[0].split('.')[0], im[1].split('.')[0]

            im_l = cv2.imread(os.path.join(data_path, im[0]))
            im_r = cv2.imread(os.path.join(data_path, im[1]))

            # Check if there's a 'result' directory
            # dir_path = config.dir_path
            dir_path = r'.'

            results_path = os.path.join(dir_path, 'results')
            if not os.path.exists(results_path):
                os.mkdir(results_path)

            # Get image points if there's none
            pts_path_l = os.path.join(dir_path, f'{im_name_l}_pts_{s}.npy')
            pts_path_r = os.path.join(dir_path, f'{im_name_r}_pts_{s}.npy')
            if not os.path.exists(pts_path_l) or not os.path.exists(pts_path_r):
                utilities.get_image_pts(im_l, im_r, pts_path_l, pts_path_r, s=s)

            # Loading picked points
            pts_left, pts_right = utilities.load_pts(f'{im_name_l}_pts_{s}', f'{im_name_r}_pts_{s}')

            # Asserting equality
            assert len(pts_left) == len(pts_right)

            if s == 's1':
                # Compute fundamental matrix using 8 point algorithm
                F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_8POINT)

            # Select only inlier points
            pts_left = pts_left[mask.ravel() == 1]
            pts_right = pts_right[mask.ravel() == 1]

            # activate fundamental matrix
            # left lines will be calculated using right image points and plotted on left image, and vice versa
            lines_left = []
            lines_right = []
            for pl, pr in zip(pts_left, pts_right):
                lines_right.append(np.dot(F, (pl[0], pl[1], 1)))
                lines_left.append(np.dot(np.transpose(F), (pr[0], pr[1], 1)))

            # draw lines
            image_l, colors = draw_lines(im_l, lines_left, pts_left)
            image_r, _ = draw_lines(im_r, lines_right, pts_right, colors)

            # Calculate Epipolar distance (SED - Symmetric Epipolar Distance)
            sed_distance = epipolar_distance(pts_left, pts_right, lines_left, lines_right)
            print(f'First images set ({im[0].split("_")[0]}) for points set {s}, has a SED result of: {sed_distance}')

            fig = plt.figure()
            fig.suptitle(f'SED = {sed_distance} (for set S{s[1]})')
            plt.subplot(121), plt.imshow(image_l)
            plt.subplot(122), plt.imshow(image_r)

            plt.savefig(fr'.\\results\\{im[0].split("_")[0]}_epipolar_lines_{s}')

            plt.show()
