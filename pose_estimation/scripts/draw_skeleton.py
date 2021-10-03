# ------------------------------------------------------------------------------
# Copyright (c) 
# Author: Yan
# Data:2021
# ------------------------------------------------------------------------------

import cv2

def draw_skeleton(img, all_key_points):
    image_skeleton = img.copy()
    POSE_COCO_PAIRS = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [5, 11], [11, 13],
                       [13, 15], [6, 8], [8, 10], [6, 12], [12, 14], [14, 16]]

    color_head = [255, 0, 0]
    color_left_arm = [255, 255, 0]
    color_left_leg = [0, 0, 255]
    color_right_arm = [0, 255, 128]
    color_right_leg = [255, 0, 255]
    color_shoulder = [204, 102, 0]

    for key_points in all_key_points:
        for pair_id, pair in enumerate(POSE_COCO_PAIRS):
            i = pair[0]
            j = pair[1]
            pt1 = (key_points[i][0], key_points[i][1])
            pt2 = (key_points[j][0], key_points[j][1])
            if 0 <= pair_id <= 3:
                color = color_head
            elif pair_id == 4:
                color = color_left_arm
            elif 5 <= pair_id <= 6:
                color = color_left_leg
            elif 7 <= pair_id <= 9:
                color = color_right_arm
            elif 10 <= pair_id <= 11:
                color = color_right_leg
            else:
                color = color_shoulder
            cv2.line(image_skeleton, pt1, pt2, color, thickness=2)

    return image_skeleton