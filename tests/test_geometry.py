from geometry import are_three_boxes_close, box_distance, is_motorcycle_near_any_person


def test_gap_between_boxes():
    assert box_distance((0, 0, 10, 10), (30, 0, 40, 10)) == 20


def test_overlapping_boxes_have_no_gap():
    assert box_distance((0, 0, 10, 10), (5, 5, 15, 15)) == 0


def test_three_riders_within_the_default_threshold():
    boxes = [(0, 0, 10, 10), (12, 0, 22, 10), (0, 12, 10, 22)]
    assert are_three_boxes_close(boxes, threshold=20)


def test_spread_out_people_are_not_a_triple():
    boxes = [(0, 0, 10, 10), (100, 0, 110, 10), (0, 100, 10, 110)]
    assert not are_three_boxes_close(boxes, threshold=20)


def test_motorcycle_must_sit_near_a_person():
    motorcycle = [(0, 0, 10, 10)]
    close_people = [(12, 0, 20, 10)]
    far_people = [(80, 80, 100, 100)]
    assert is_motorcycle_near_any_person(motorcycle, close_people, threshold=20)
    assert not is_motorcycle_near_any_person(motorcycle, far_people, threshold=20)
