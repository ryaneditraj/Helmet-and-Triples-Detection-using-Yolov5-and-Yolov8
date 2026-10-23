import os
import time
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

from config import camera_source, load_dotenv

load_dotenv()


def create_session_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"static/no_helmet/NO_HELMET_session_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_snapshot(frame, folder):
    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    filename = os.path.join(folder, f"snapshot_{timestamp}.jpg")
    success = cv2.imwrite(filename, frame)
    if success:
        print(f"📸 Saved: {filename}")
    else:
        print("❌ Failed to save snapshot.")


# Load YOLOv8 model
model = YOLO(os.environ.get("HELMET_WEIGHTS", "best.pt"))

# Open webcam
cap = cv2.VideoCapture(camera_source())
if not cap.isOpened():
    print("❌ Webcam could not be opened.")
    raise SystemExit(1)

names = model.names  # class id to label mapping
session_folder = None
session_active = False
session_end_time = None
last_snapshot_time = 0
snapshot_interval = 0.25  # seconds
session_duration = 10  # seconds

print("🎥 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    raw_frame = frame.copy()
    results = model(frame)
    boxes = results[0].boxes
    detected_classes = [names[int(cls)] for cls in boxes.cls]

    if "Without Helmet" in detected_classes:
        now = datetime.now()

        # Start session if not already active
        if not session_active or now > session_end_time:
            session_folder = create_session_folder()
            session_end_time = now + timedelta(seconds=session_duration)
            session_active = True
            print(f"🚨 New session started: {session_folder}")

        # During active session, save image every 250ms
        if current_time - last_snapshot_time >= snapshot_interval:
            save_snapshot(raw_frame, session_folder)
            last_snapshot_time = current_time

    else:
        if session_active and datetime.now() > session_end_time:
            session_active = False
            session_folder = None
            print("✅ Session ended.\n")

    # Show annotated preview
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
