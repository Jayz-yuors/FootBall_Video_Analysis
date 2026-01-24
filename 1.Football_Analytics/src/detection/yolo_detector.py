# src/detection/yolo_detector.py
import os
from ultralytics import YOLO
from src.utils.logger import log
from src.utils.config import config
class YOLODetector:
    def __init__(self):
        """
        Load YOLO model in the safest possible way:
        1. First try using your local project models/yolo/yolov8n.pt
        2. If missing → fallback to YOLO('yolov8n.pt')
           which auto-downloads model from Ultralytics cache.
        """
        log("Initializing YOLOv8 model...")

        # Expected local path for YOLOv8n
        local_model_path = config.YOLO_MODEL  # models/yolo/yolov8n.pt

        # Check if local model exists
        if os.path.exists(local_model_path):
            log(f"Local YOLO model found at: {local_model_path}")
            self.model = YOLO(local_model_path)

        else:
            log(f"No local YOLO model found at: {local_model_path}")
            log("Falling back to YOLO('yolov8n.pt') → auto-download will occur.")
            self.model = YOLO("yolov8n.pt")  # Ultralytics auto-download

        log("YOLO model loaded successfully.")

    def detect_frame(self, frame):
        """
        Perform YOLO detection on a single frame.
        Returns:
            - annotated frame (with boxes drawn)
            - detection results object
        """
        results = self.model(frame)
        annotated = results[0].plot()  # YOLO auto-draws bounding boxes
        return annotated, results
