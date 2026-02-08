# src/analytics/heatmap.py

import cv2
import numpy as np
import os
from typing import Dict, List


def draw_pitch_background(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Green pitch
    frame[:] = (40, 140, 40)

    white = (240, 240, 240)
    thickness = 2

    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)

    x1, y1 = margin_x, margin_y
    x2, y2 = width - margin_x, height - margin_y

    # Outer lines
    cv2.rectangle(frame, (x1, y1), (x2, y2), white, thickness)

    # Halfway line
    mid_x = (x1 + x2) // 2
    cv2.line(frame, (mid_x, y1), (mid_x, y2), white, thickness)

    # Center circle
    center = (mid_x, (y1 + y2) // 2)
    radius = int((y2 - y1) * 0.18)
    cv2.circle(frame, center, radius, white, thickness)
    cv2.circle(frame, center, 4, white, -1)

    return frame


def generate_team_heatmaps(
    tracking_log: Dict,
    frame_width: int,
    frame_height: int,
    output_dir: str,
):
    """
    Creates heatmap images for Team A and Team B.
    Uses stored tracking history ONLY.
    """

    os.makedirs(output_dir, exist_ok=True)

    heatmaps = {
        "A": np.zeros((frame_height, frame_width), dtype=np.float32),
        "B": np.zeros((frame_height, frame_width), dtype=np.float32),
    }

    # Accumulate positions
    for track_id, data in tracking_log.items():
        team = data.get("team")
        if team not in ("A", "B"):
            continue

        for _, x1, y1, x2, y2 in data["history"]:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if 0 <= cx < frame_width and 0 <= cy < frame_height:
                heatmaps[team][cy, cx] += 1.0

    # Generate images
    for team, heat in heatmaps.items():
        if np.max(heat) == 0:
            continue

        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=35, sigmaY=35)
        heat = heat / np.max(heat)
        heat = np.uint8(255 * heat)

        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        pitch = draw_pitch_background(frame_width, frame_height)

        overlay = cv2.addWeighted(pitch, 0.6, heat_color, 0.7, 0)

        out_path = os.path.join(output_dir, f"heatmap_team_{team}.png")
        cv2.imwrite(out_path, overlay)

        print(f"[HEATMAP] Saved → {out_path}")
