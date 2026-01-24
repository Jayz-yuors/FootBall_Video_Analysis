# src/analytics/possession.py

import math
from collections import Counter, deque
from typing import Dict, List, Tuple, Optional


class PossessionTracker:
    """
    Advanced possession estimator for Team A and Team B.

    Logic per frame:

    1. If the ball is visible:
       - Compute ball center.
       - For every tracked player with team label ("A"/"B"):
         * Compute distance to ball.
         * Estimate player motion (velocity) from previous frame.
         * Give higher score if:
             - Player is closer to the ball.
             - Player is moving TOWARDS the ball.
       - Choose player with highest score above a threshold as the
         "winner" of that frame.

    2. If no clear winner:
       - Use short-term memory of last controller (a few frames),
         so possession doesn’t flicker randomly.

    3. Possession displayed is based on a sliding window of the last
       N seconds of frame winners (mildly smoothed but still dynamic).
    """

    def __init__(
        self,
        fps: float,
        radius: float = 60.0,
        window_seconds: float = 3.0,   # shorter window → more dynamic
        memory_seconds: float = 0.5,   # short lock on last controller
    ) -> None:
        """
        :param fps: video FPS (for converting seconds -> frames)
        :param radius: base distance in pixels to consider players
                       for ball control.
        :param window_seconds: smoothing window for overlay (seconds)
        :param memory_seconds: how long (seconds) last controller
                               can keep possession when ball is
                               unclear / temporarily lost.
        """
        self.radius = radius
        self.fps = fps
        self.window_frames = max(1, int(window_seconds * fps))
        self.memory_frames = max(1, int(memory_seconds * fps))

        # Sliding window of recent frame winners: "A", "B" or None
        self.recent_winners: deque[Optional[str]] = deque(
            maxlen=self.window_frames
        )

        # Global counts over the whole video (optional stats)
        self.global_counts: Counter = Counter()
        self.total_ball_frames: int = 0

        # For motion estimation (per-track last position)
        self.prev_player_centers: Dict[int, Tuple[float, float]] = {}

        # Short-term controller memory
        self.last_team: Optional[str] = None
        self.last_team_memory_left: int = 0  # frames remaining

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ball_center(
        self, ball_box: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = ball_box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _player_centers(
        self,
        tracks: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> Dict[int, Tuple[float, float]]:
        centers: Dict[int, Tuple[float, float]] = {}
        for track_id, box in tracks:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers[track_id] = (cx, cy)
        return centers

    # ------------------------------------------------------------------ #
    # Core update logic
    # ------------------------------------------------------------------ #

    def update(
        self,
        tracks: List[Tuple[int, Tuple[float, float, float, float]]],
        frame_team_map: Dict[int, str],
        ball_box: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """
        Update possession based on a single frame.

        :param tracks: list of (track_id, (x1, y1, x2, y2)) for this frame
        :param frame_team_map: {track_id: "A" or "B"} for valid players
        :param ball_box: (x1, y1, x2, y2) for ball, or None if not detected
        """

        # Pre-compute player centers + velocities for this frame
        player_centers = self._player_centers(tracks)

        # Estimate velocities from previous centers
        player_velocities: Dict[int, Tuple[float, float]] = {}
        for track_id, (cx, cy) in player_centers.items():
            if track_id in self.prev_player_centers:
                px, py = self.prev_player_centers[track_id]
                player_velocities[track_id] = (cx - px, cy - py)
            else:
                player_velocities[track_id] = (0.0, 0.0)

        # Update stored centers for next frame
        self.prev_player_centers = player_centers

        winner_team: Optional[str] = None
        observed_directly = False  # only True if we had a real detection

        # ---------------- Ball visible ---------------- #
        if ball_box is not None:
            ball_cx, ball_cy = self._ball_center(ball_box)

            best_score = 0.0
            best_team = None

            for track_id, (px, py) in player_centers.items():
                team = frame_team_map.get(track_id)
                if team not in ("A", "B"):
                    continue

                dist = math.hypot(ball_cx - px, ball_cy - py)
                if dist > self.radius:
                    # Too far to realistically control
                    continue

                # Distance score: closer → higher (0..1)
                dist_score = max(0.0, (self.radius - dist) / self.radius)

                # Motion score: moving toward ball → bonus
                vx, vy = player_velocities.get(track_id, (0.0, 0.0))
                to_ball_x = ball_cx - px
                to_ball_y = ball_cy - py
                # Dot product between velocity and direction to ball
                dot = vx * to_ball_x + vy * to_ball_y
                motion_score = 0.0
                if dot > 0:
                    # normalize a bit to keep in [0,1] range
                    motion_score = 0.3  # fixed small bonus if moving toward

                # Final score: mostly distance, small motion bonus
                score = dist_score + motion_score

                if score > best_score:
                    best_score = score
                    best_team = team

            # Threshold for "clear" control
            if best_team is not None and best_score >= 0.25:
                winner_team = best_team
                observed_directly = True

        # ---------------- If unclear, use memory ---------------- #
        if winner_team is None:
            # Decrease memory
            if self.last_team_memory_left > 0 and self.last_team in ("A", "B"):
                winner_team = self.last_team
                self.last_team_memory_left -= 1
            else:
                winner_team = None

        # ---------------- Update state ---------------- #
        # Update memory if we actually saw a clear winner this frame
        if observed_directly and winner_team in ("A", "B"):
            self.last_team = winner_team
            self.last_team_memory_left = self.memory_frames

        # Add to sliding window
        self.recent_winners.append(winner_team)

        # Update global stats only when ball exists and we had clear winner
        if observed_directly and winner_team in ("A", "B"):
            self.global_counts[winner_team] += 1
            self.total_ball_frames += 1

    # ------------------------------------------------------------------ #
    # Percentage helpers
    # ------------------------------------------------------------------ #

    def _window_percentages(self) -> Dict[str, float]:
        """
        Compute possession from the sliding window only.
        """
        window_counts = Counter(t for t in self.recent_winners if t in ("A", "B"))
        total = sum(window_counts.values())
        if total == 0:
            return {"A": 0.0, "B": 0.0}
        return {
            "A": 100.0 * window_counts.get("A", 0) / total,
            "B": 100.0 * window_counts.get("B", 0) / total,
        }

    def _global_percentages(self) -> Dict[str, float]:
        """
        Possession over the entire video so far.
        """
        if self.total_ball_frames == 0:
            return {"A": 0.0, "B": 0.0}
        return {
            "A": 100.0 * self.global_counts.get("A", 0) / self.total_ball_frames,
            "B": 100.0 * self.global_counts.get("B", 0) / self.total_ball_frames,
        }

    def get_percentages(self) -> Dict[str, float]:
        """
        Public accessor — returns window-based percentages.
        If window has no info yet, falls back to global stats.
        """
        win = self._window_percentages()
        if (win["A"] + win["B"]) == 0:
            return self._global_percentages()
        return win

    # ------------------------------------------------------------------ #
    # Drawing overlay (bottom-right)
    # ------------------------------------------------------------------ #

    def draw_overlay(self, frame) -> None:
        """
        Draw possession overlay on the bottom-right corner of the frame.
        Uses smoothed window percentages.
        """
        import cv2  # local import

        h, w = frame.shape[:2]
        stats = self.get_percentages()

        text_line1 = f"Team A Possession: {stats['A']:.1f}%"
        text_line2 = f"Team B Possession: {stats['B']:.1f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Measure text sizes
        (w1, h1), _ = cv2.getTextSize(text_line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(text_line2, font, font_scale, thickness)

        box_width = max(w1, w2) + 20
        box_height = h1 + h2 + 20

        x2 = w - 10
        y2 = h - 10
        x1 = x2 - box_width
        y1 = y2 - box_height

        # Background rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Draw text lines
        cv2.putText(
            frame,
            text_line1,
            (x1 + 10, y1 + 10 + h1),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text_line2,
            (x1 + 10, y1 + 10 + h1 + h2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
