# ------------------------------------------------------------------------------
# Copyright (c)
# Author: Yan
# Data:2021
# ------------------------------------------------------------------------------

import json

def remove_sqaure_brackets(person_points_2d):
    person_points_2d = person_points_2d.tolist()
    person_points_2d_list = []
    for key_point in person_points_2d:
        for item in key_point:
            person_points_2d_list.append(item)

        #person_points_2d_list.append('\n')

    return person_points_2d_list

def write_keypoint_json(count, json_file_path, pose_preds, version):
    all_persons = []
    for person_id, person_points_2d in enumerate(pose_preds):
        person_points_2d_list = remove_sqaure_brackets(person_points_2d)
        each_person = {"person_id": person_id, "pose_keypoints_2d": person_points_2d_list}

        all_persons.append(each_person)

    json_data = {"people": all_persons, "version": version}

    with open(json_file_path, 'w') as f:
        json_formated_data = json.dumps(json_data, indent=4)
        json.dump(json_data, f, indent=4, separators=(',', ':'))

    return person_points_2d.tolist()