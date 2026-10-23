from itertools import combinations

import numpy as np


def box_distance(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return np.hypot(dx, dy)


def are_three_boxes_close(boxes, threshold=20):
    for triplet in combinations(boxes, 3):
        dists = [
            box_distance(triplet[0], triplet[1]),
            box_distance(triplet[0], triplet[2]),
            box_distance(triplet[1], triplet[2]),
        ]
        if all(d <= threshold for d in dists):
            return True
    return False


def is_motorcycle_near_any_person(motorcycle_boxes, person_boxes, threshold=20):
    for motorcycle_box in motorcycle_boxes:
        for person_box in person_boxes:
            if box_distance(motorcycle_box, person_box) <= threshold:
                return True
    return False
