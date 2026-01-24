# src/analytics/team_classifier.py

import cv2
import numpy as np
from typing import Tuple, Optional


class TeamClassifier:
    """
    Very simple, robust color-based team classifier.

    - Looks only at the TORSO area of the bounding box (ignore grass, shorts).
    - Uses HSV color masks for:
        * Red jerseys  -> Team A
        * Blue jerseys -> Team B
        * Yellow (referee) -> ignored
    - Returns: "A", "B" or None (ignore / unknown).
    """

    def __init__(self) -> None:
        # HSV ranges tuned for broadcast footage (you can tweak later)
        # Red jersey (Manchester United)
        self.red_lower1 = np.array([0, 80, 60], dtype=np.uint8)
        self.red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([160, 80, 60], dtype=np.uint8)
        self.red_upper2 = np.array([179, 255, 255], dtype=np.uint8)

        # Blue jersey (Everton)
        self.blue_lower = np.array([95, 80, 60], dtype=np.uint8)
        self.blue_upper = np.array([135, 255, 255], dtype=np.uint8)

        # Referee yellow (to ignore)
        self.yellow_lower = np.array([20, 70, 70], dtype=np.uint8)
        self.yellow_upper = np.array([35, 255, 255], dtype=np.uint8)

    def _crop_torso(self, frame, box: Tuple[float, float, float, float]):
        """
        Crop approximate torso region from the bounding box.
        We ignore head/legs to avoid grass/shorts influence.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)

        # Clamp to frame
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame - 1))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        h = y2 - y1
        # Torso: from 25% to 75% of bbox height
        top = y1 + int(0.25 * h)
        bottom = y1 + int(0.75 * h)
        top = max(y1, min(top, y2 - 1))
        bottom = max(top + 1, min(bottom, y2))

        torso = frame[top:bottom, x1:x2]
        if torso.size == 0:
            return None
        return torso

    def classify(self, frame, box: Tuple[float, float, float, float]) -> Optional[str]:
        """
        Classify a single player box into Team A, Team B, or None.

        Returns:
            "A"  -> red team
            "B"  -> blue team
            None -> referee / crowd / unknown
        """
        torso = self._crop_torso(frame, box)
        if torso is None:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # Red mask (two ranges)
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_count = int(cv2.countNonZero(mask_red))

        # Blue mask
        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_count = int(cv2.countNonZero(mask_blue))

        # Yellow (referee) mask
        mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        yellow_count = int(cv2.countNonZero(mask_yellow))

        # No strong color
        if max(red_count, blue_count, yellow_count) < 20:
            return None

        # Ignore ref
        if yellow_count == max(red_count, blue_count, yellow_count):
            return None

        # Decide team
        if red_count > blue_count:
            return "A"
        elif blue_count > red_count:
            return "B"
        else:
            return None
