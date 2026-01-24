import cv2
import os
from .helpers import ensure_dir
from .logger import log
from .config import config
def extract_frames(video_path,step = 1):
    ensure_dir(config.FRAMES_DIR)
    cap = cv2.VideoCapture(video_path)
    count = 0 
    saved = 0
    while True :
        ret , frame = cap.read()
        if not ret :
            break 
        if count % step == 0 :
            frames_path = os.path.join(config.FRAMES_DIR,f"frame_{saved}.jpg")
            cv2.imwrite(frames_path,frame)
            saved +=1
        count +=1
        cap.release()
        log(f"Extracted {saved} frames from video.")
