# run_pipeline.py
import os 
from src.pipeline.pipeline import Pipeline
from src.utils.config import config

if __name__ == "__main__":
    video_path = os.path.join(config.SAMPLE_VIDEOS, "sample1.mp4")

    pipeline = Pipeline(video_path)
    pipeline.process_video()
