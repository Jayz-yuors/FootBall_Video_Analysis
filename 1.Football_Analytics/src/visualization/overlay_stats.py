# src/visualization/overlay_stats.py

import cv2
from typing import Dict
def overlay_match_stats(
    frame,
    stats: Dict[str, float],
    start_x: int = 10,
    start_y: int = 30,
    line_height: int = 25,
):
    """
    Overlay text stats on the top-left corner of the frame.
    stats: {"TeamA_possession": 60.5, "TeamB_possession": 39.5, ...}
    """
    y = start_y
    for key, value in stats.items():
        text = f"{key}: {value:.1f}"
        cv2.putText(
            frame,
            text,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += line_height
    return frame
