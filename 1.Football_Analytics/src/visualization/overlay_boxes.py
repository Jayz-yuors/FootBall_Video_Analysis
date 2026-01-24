import cv2
from typing import List, Tuple

TEAM_COLORS = {
    "A": (0, 0, 255),    # Red (BGR)
    "B": (255, 0, 0),    # Blue (BGR)
    "Unknown": (128, 128, 128),  # Grey
}


def draw_boxes_with_team(
        frame,
        tracks: List[Tuple[int, Tuple[float, float, float, float]]],
        team_classifier
):
    for track_id, box in tracks:
        x1, y1, x2, y2 = map(int, box)

        # Which team?
        team = team_classifier.get_team(track_id)

        # Select color
        color = TEAM_COLORS.get(team, TEAM_COLORS["Unknown"])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Write label text
        label = f"{team} | ID {track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame
