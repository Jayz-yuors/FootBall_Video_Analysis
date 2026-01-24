# src/visualization/ball_marker.py

import cv2
from typing import Tuple, Optional


def draw_ball_marker(frame, box: Optional[Tuple[float, float, float, float]]) -> None:
    """
    Draw a dark maroon box around the detected ball.

    - Box is small and only around the ball.
    - No trails, no extra effects.
    """
    if box is None:
        return

    x1, y1, x2, y2 = map(int, box)

    # Dark maroon-ish color (BGR)
    color = (32, 0, 128)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
