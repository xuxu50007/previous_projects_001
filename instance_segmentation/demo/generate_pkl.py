# ----------------------------------
# Copyright (c)
# Modified by Yan
# Date: 2021
# ----------------------------------
import numpy as np
import cv2
import pickle


def generate_pkl(seg_result, score_th, only_person=False):
    # generate bounding boxes
    bboxes = []
    for idx, item in enumerate(seg_result[0]):
        if idx <= len(seg_result[0]) - 1:
            for seg in item.cpu().numpy():
                if isinstance(seg, np.ndarray):
                    seg == np.array(seg)
                    y_index = []
                    for i in range(seg.shape[0]):
                        if 1 in seg[i, :]:
                            y_index.append(i)
                    x_index = []
                    for i in range(seg.shape[1]):
                        if 1 in seg[:, i]:
                            x_index.append(i)
                    bboxes.append([y_index[0], x_index[0], y_index[-1], x_index[-1]])

    mask = seg_result[0][0].cpu().numpy()
    rois = bboxes
    class_ids = seg_result[0][1].cpu().numpy().tolist()
    scores = seg_result[0][2].cpu().numpy().tolist()

    mask_filtered = []
    rois_filtered = []
    class_ids_filtered = []
    scores_filtered = []
    for i in range(len(scores)):
        if scores[i] >= score_th:
            # person is 0
            if only_person and class_ids[i] == 0:
                mask_filtered.append(mask[i])
                rois_filtered.append(rois[i])
                class_ids_filtered.append(class_ids[i])
                scores_filtered.append(scores[i])

    mask = np.dstack(mask_filtered)
    rois = np.array(rois_filtered)
    scores = np.array(scores_filtered)

    # Add 1 to classId to account for BG as class 0 in Mask-RCNN
    class_ids = np.array(class_ids_filtered) + 1

    # generate a dictionary to store segmentation info
    result_dict = {"masks": mask, "rois": rois, "class_ids": class_ids, "scores": scores}

    return result_dict