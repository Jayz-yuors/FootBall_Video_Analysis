import cv2
from typing import Tuple

# BGR colors
COLOR_TEAM_A = (0, 0, 255)   # Red
COLOR_TEAM_B = (255, 0, 0)   # Blue


def draw_player_marker(frame, team: str, box: Tuple[float, float, float, float]):
    """
    Draw a clean box + single-letter label for a player.

    team: "A" or "B"
    box: (x1, y1, x2, y2) in image coordinates
    """
    x1, y1, x2, y2 = map(int, box)

    if team == "A":
        color = COLOR_TEAM_A
        label = "A"
    else:
        color = COLOR_TEAM_B
        label = "B"

    # Smooth rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    label_x = x1
    label_y = max(y1 - 5, th + 2)

    cv2.rectangle(
        frame,
        (label_x, label_y - th - 4),
        (label_x + tw + 4, label_y + 2),
        color,
        thickness=-1,
    )

    cv2.putText(
        frame,
        label,
        (label_x + 2, label_y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )