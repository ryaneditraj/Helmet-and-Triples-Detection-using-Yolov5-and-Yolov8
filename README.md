# Helmet-and-Triples-Detection-using-Yolov5-and-Yolov8
This project uses YOLOv8 for detecting riders without helmets and YOLOv5 for detecting triple riding.   It saves snapshots of each offence in timestamped folders and displays them on a Flask dashboard.
```mermaid
flowchart LR
    A[Live Video Feed] --> B[YOLOv8: Helmet Detection]
    B --> C[YOLOv5: Motorcycle & Person Detection]
    C --> D[Violation Capture: Image + Timestamp]
    D --> E[Police Dashboard Review]
