import os
from datetime import datetime, timedelta

import cv2

from config import camera_source, load_dotenv
from geometry import are_three_boxes_close, is_motorcycle_near_any_person
from yolo_loader import load_yolov5

load_dotenv()


def get_boxes_by_class(results, target_class):
    boxes = []
    detections = results.pred[0]
    for det in detections:
        class_id = int(det[5])
        if results.names[class_id] == target_class:
            x1, y1, x2, y2 = map(int, det[:4])
            boxes.append((x1, y1, x2, y2))
    return boxes


def create_session_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"session_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_snapshot(raw_frame, folder):
    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    filename = os.path.join(folder, f"snapshot_{timestamp}.jpg")
    success = cv2.imwrite(filename, raw_frame)
    if success:
        print(f"📸 Saved: {filename}")
    else:
        print("❌ Failed to save snapshot.")


def detect_from_webcam(model, source=1, box_distance_threshold=20):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    session_folder = None
    session_end_time = None
    print("🎥 Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Make a copy BEFORE passing to model
        raw_frame = frame.copy()

        # AI processing on a separate copy
        results = model(frame)

        # Get detection results
        person_boxes = get_boxes_by_class(results, "person")
        motorcycle_boxes = get_boxes_by_class(results, "motorcycle")
        person_count = len(person_boxes)

        # Snapshot condition check
        condition = (
            person_count >= 3
            and are_three_boxes_close(person_boxes, box_distance_threshold)
            and len(motorcycle_boxes) >= 1
            and is_motorcycle_near_any_person(
                motorcycle_boxes, person_boxes, box_distance_threshold
            )
        )

        now = datetime.now()
        if condition:
            if session_folder is None or now > session_end_time:
                session_folder = create_session_folder()
                session_end_time = now + timedelta(seconds=10)
                print(f"🗂️ New session started: {session_folder}")

            save_snapshot(raw_frame, session_folder)  # Only save untouched raw frame
        else:
            if session_end_time is not None and now > session_end_time:
                session_folder = None
                session_end_time = None

        # Show annotated frame for user (optional)
        results.render()
        annotated = results.ims[0]
        cv2.imshow("AI Dashcam - YOLOv5", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_yolov5()
    # Keep device 1 as the historical default. CAMERA_SOURCE overrides it.
    source = camera_source() if os.environ.get("CAMERA_SOURCE") else 1
    detect_from_webcam(model, source=source, box_distance_threshold=20)
