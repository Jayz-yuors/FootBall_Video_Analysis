import cv2
import os
import json
from .logger import log
from .helpers import ensure_dir
from .config import config
import hashlib 
def load_video(video_path : str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    log(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    metadata = {
        "fps":cap.get(cv2.CAP_PROP_FPS), #internal functions of cvv2
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    metadata["duration_sec"] = (
        metadata["frame_count"] / metadata["fps"]
        if metadata["fps"] > 0 else 0
    )
    save_metadata(video_path, metadata)
    return cap, metadata
def save_metadata(video_path : str , metadata : dict):
    ensure_dir(config.METADATA_DIR)
    hashed = hashlib.md5(video_path.encode()).hexdigest() + ".json" # a unique JSON name
    metadata_file = os.path.join(config.METADATA_DIR, hashed)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    log(f"Metadata saved → {metadata_file}")

