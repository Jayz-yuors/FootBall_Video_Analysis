# src/utils/config.py

import os


class Config:
    # ---------- Base directories ----------
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VIDEO_DIR = os.path.join(DATA_DIR, "videos")
    USER_UPLOADS = os.path.join(VIDEO_DIR, "user_uploads")
    SAMPLE_VIDEOS = os.path.join(VIDEO_DIR, "samples")

    METADATA_DIR = os.path.join(DATA_DIR, "metadata")
    FRAMES_DIR = os.path.join(DATA_DIR, "frames")
    TRACKING_LOGS_DIR = os.path.join(DATA_DIR, "tracking_logs")

    # ---------- Output directories ----------
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    OUTPUT_VIDEOS = os.path.join(OUTPUT_DIR, "videos")
    OUTPUT_STATS = os.path.join(OUTPUT_DIR, "stats")
    OUTPUT_HEATMAPS = os.path.join(OUTPUT_DIR, "heatmaps")

    # ---------- Model paths ----------
    YOLO_MODEL = os.path.join(BASE_DIR, "models", "yolo", "yolov8n.pt")

    # ---------- Tracking / logging ----------
    SAVE_TRACKING_LOGS = True

    # ---------- Ball detection ----------
    # COCO 'sports ball' class id
    BALL_CLASS_ID = 32
    BALL_CONF_THRESH = 0.35
    BALL_TRAIL_HISTORY = 15  # number of points in the trail


config = Config()
