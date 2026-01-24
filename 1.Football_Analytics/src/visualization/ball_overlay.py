# src/visualization/ball_overlay.py

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_ball_overlay(
    frame,
    center: Optional[Tuple[int, int]],
    trail_points: List[Tuple[int, int]],
    color=(0, 255, 0),
):
    """
    Draws a small green triangle pointer above the ball + a smooth trail.

    :param frame: BGR image
    :param center: (cx, cy) of ball center or None
    :param trail_points: list of previous centers for trail
    :param color: BGR color for pointer + trail
    """
    if center is None:
        return frame

    cx, cy = center

    # ---------- Draw small triangle pointer (subtle - Option A) ----------
    pointer_height = 12
    pointer_width = 16
    offset = 10  # distance above ball center

    tip = (cx, max(cy - offset - pointer_height, 0))
    left = (cx - pointer_width // 2, max(cy - offset, 0))
    right = (cx + pointer_width // 2, max(cy - offset, 0))

    pts = np.array([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color)

    # ---------- Draw curved trail ----------
    if len(trail_points) >= 2:
        pts_trail = np.array(trail_points, dtype=np.int32)
        # Slightly thinner line to look clean
        cv2.polylines(frame, [pts_trail], isClosed=False, color=color, thickness=2)

    return frame
