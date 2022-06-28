import cv2
import numpy as np


def draw_lines(img, lines, points, picked_colors=None):
    r, c, _ = img.shape
    picked_colors_list = []
    for idx, (r, point) in enumerate(zip(lines, points)):
        if picked_colors is None:
            color = tuple(np.random.randint(0, 255, 3).tolist())

            while color in picked_colors_list:
                color = tuple(np.random.randint(0, 255, 3).tolist())

            picked_colors_list.append(color)

        else:
            color = picked_colors[idx]

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv2.circle(img, tuple(int(p) for p in point), 5, color, -1)

    return img, picked_colors_list


def epipolar_distance(l_pts, r_pts, l_lines, r_lines):
    distance = 0

    for l_pt, r_pt, l_line, r_line in zip(l_pts, r_pts, l_lines, r_lines):
        d_left = np.dot((l_pt[0], l_pt[1], 1), l_line) ** 2 / (l_line[0] ** 2 + l_line[1] ** 2)
        d_right = np.dot((r_pt[0], r_pt[1], 1), r_line) ** 2 / (r_line[0] ** 2 + r_line[1] ** 2)
        distance += np.sqrt(d_left + d_right)

    distance /= len(l_pts)

    return distance
