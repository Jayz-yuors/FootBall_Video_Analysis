# src/visualization/minimap.py

import cv2
import numpy as np
from typing import List, Dict, Optional


def create_minimap_frame(
    width: int,
    height: int,
    players: List[Dict],
    ball: Optional[Dict] = None,
) -> np.ndarray:
    """
    Create a single minimap frame.

    players: list of dicts:
        {
            "team": "A" or "B",
            "id":   int track id,
            "cx":   float in [0, 1]  # normalized X (0 = left, 1 = right)
            "cy":   float in [0, 1]  # normalized Y (0 = top, 1 = bottom)
        }

    ball: optional dict:
        {
            "cx": float in [0, 1],
            "cy": float in [0, 1],
        }
    """

    # --- 1. Green modern field background (Option A) ---
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Two shades of green stripes
    stripe_colors = [(40, 120, 40), (50, 160, 50)]  # BGR
    stripe_count = 8
    stripe_width = width // stripe_count

    for i in range(stripe_count):
        x1 = i * stripe_width
        x2 = width if i == stripe_count - 1 else (i + 1) * stripe_width
        color = stripe_colors[i % 2]
        cv2.rectangle(frame, (x1, 0), (x2, height), color, thickness=-1)

    # Midline and center circle (for aesthetics)
    mid_x = width // 2
    cv2.line(frame, (mid_x, 0), (mid_x, height), (230, 230, 230), 1)

    center = (width // 2, height // 2)
    pitch_radius = int(min(width, height) * 0.18)
    cv2.circle(frame, center, pitch_radius, (230, 230, 230), 1)
    cv2.circle(frame, center, 2, (230, 230, 230), -1)

    # --- 2. Draw players ---
    for p in players:
        team = p.get("team")
        tid = p.get("id")
        cx_norm = float(p.get("cx", 0.5))
        cy_norm = float(p.get("cy", 0.5))

        x = int(cx_norm * width)
        y = int(cy_norm * height)

        # Colors for teams
        if team == "A":
            color = (0, 0, 255)    # Red (BGR)
        else:
            color = (255, 0, 0)    # Blue (BGR)

        # Player dot
        cv2.circle(frame, (x, y), 6, color, thickness=-1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), thickness=1)

        # Small ID text above the dot
        label = str(tid)
        cv2.putText(
            frame,
            label,
            (x - 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # --- 3. Draw ball (if visible) ---
    if ball is not None:
        bx = int(float(ball.get("cx", 0.5)) * width)
        by = int(float(ball.get("cy", 0.5)) * height)
        # Bright yellow for ball
        cv2.circle(frame, (bx, by), 5, (0, 255, 255), thickness=-1)
        cv2.circle(frame, (bx, by), 5, (0, 0, 0), thickness=1)

    return frame
