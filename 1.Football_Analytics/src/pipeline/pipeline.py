# src/pipeline/pipeline.py

import os
import json
import cv2

from src.utils.video_loader import load_video
from src.detection.yolo_detector import YOLODetector
from src.utils.logger import log
from src.utils.config import config
from src.utils.helpers import ensure_dir
from src.tracking.tracker import SimpleTracker
from src.analytics.team_classifier import TeamClassifier
from src.visualization.player_marker import draw_player_marker
from src.visualization.ball_marker import draw_ball_marker
from src.analytics.possession import PossessionTracker
from src.visualization.minimap import create_minimap_frame


class Pipeline:
    def __init__(self, video_path: str):
        log("Initializing Pipeline ::")
        self.video_path = video_path

        # Load video + metadata
        self.cap, self.metadata = load_video(video_path)
        self.detector = YOLODetector()
        self.tracker = SimpleTracker(max_distance=60.0)
        self.team_classifier = TeamClassifier()

        # Ball temporal smoothing state
        self.ball_history = []       # list of recent ball boxes or None
        self.max_ball_history = 5    # window size
        self.smoothed_ball = None    # final smoothed ball box

        # Possession tracker (Phase 5)
        fps = (
            self.metadata.get("fps", 25.0)
            if isinstance(self.metadata, dict)
            else 25.0
        )
        self.possession_tracker = PossessionTracker(fps=fps)

        ensure_dir(config.OUTPUT_VIDEOS)
        ensure_dir(config.TRACKING_LOGS_DIR)

    # ------------------------------------------------------------
    # Helper filters / ball selection / smoothing
    # ------------------------------------------------------------

    def _is_on_pitch(self, frame, box):
        """
        Prevent crowd detections from passing as players.
        """
        h_frame, _ = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1

        if h < 0.06 * h_frame:      # too small
            return False
        if y2 < 0.25 * h_frame:     # too high (stands)
            return False
        return True

    def _select_ball(self, frame, candidates):
        """
        Choose the best raw YOLO 'sports ball' candidate.

        candidates: list of (x1, y1, x2, y2, score)
        returns: (x1, y1, x2, y2) or None
        """
        if not candidates:
            return None

        h_frame, w_frame = frame.shape[:2]
        best = None
        best_score = -1.0

        for (x1, y1, x2, y2, score) in candidates:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            area = w * h
            # reject too small / too large
            if area < 4 or area > 0.02 * w_frame * h_frame:
                continue

            aspect = w / float(h)
            if aspect < 0.5 or aspect > 1.8:
                continue

            if y2 < 0.20 * h_frame:
                continue

            if score > best_score:
                best_score = score
                best = (float(x1), float(y1), float(x2), float(y2))

        return best

    def _update_ball_smoothing(self, ball_box):
        """
        Temporal smoothing & multi-frame voting logic for ball.

        - Keeps a small history of recent detections.
        - Averages center/size over valid detections.
        - If missing for several frames, clears the ball.
        """
        # append new detection (or None)
        self.ball_history.append(ball_box)
        if len(self.ball_history) > self.max_ball_history:
            self.ball_history.pop(0)

        # how many trailing misses?
        trailing_misses = 0
        for b in reversed(self.ball_history):
            if b is None:
                trailing_misses += 1
            else:
                break

        if trailing_misses >= 3:
            self.smoothed_ball = None
            return

        valid_boxes = [b for b in self.ball_history if b is not None]
        if not valid_boxes:
            return

        cx_list, cy_list, w_list, h_list = [], [], [], []

        for (x1, y1, x2, y2) in valid_boxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            cx_list.append(cx)
            cy_list.append(cy)
            w_list.append(w)
            h_list.append(h)

        avg_cx = sum(cx_list) / len(cx_list)
        avg_cy = sum(cy_list) / len(cy_list)
        avg_w = sum(w_list) / len(w_list)
        avg_h = sum(h_list) / len(h_list)

        x1_s = avg_cx - avg_w / 2.0
        y1_s = avg_cy - avg_h / 2.0
        x2_s = avg_cx + avg_w / 2.0
        y2_s = avg_cy + avg_h / 2.0

        self.smoothed_ball = (x1_s, y1_s, x2_s, y2_s)

    # ------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------

    def process_video(self):
        log("Processing video frame-by-frame ::")
        frame_idx = 0
        tracking_log = {}

        base_name = os.path.basename(self.video_path)

        output_path_main = os.path.join(
            config.OUTPUT_VIDEOS,
            f"processed_{base_name}",
        )
        output_path_minimap = os.path.join(
            config.OUTPUT_VIDEOS,
            f"minimap_{base_name}",
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.metadata["fps"] if self.metadata.get("fps", 0) > 0 else 25.0

        # Main processed video
        main_out = cv2.VideoWriter(
            output_path_main,
            fourcc,
            fps,
            (self.metadata["width"], self.metadata["height"]),
        )

        # Mini-map video (fixed smaller resolution)
        minimap_width, minimap_height = 640, 360
        minimap_out = cv2.VideoWriter(
            output_path_minimap,
            fourcc,
            fps,
            (minimap_width, minimap_height),
        )

        frame_w = float(self.metadata["width"])
        frame_h = float(self.metadata["height"])

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # --- YOLO Detection ---
            results = self.detector.model(frame)[0]
            detections = []          # player boxes
            ball_candidates = []     # sports ball boxes

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes.data:
                    x1, y1, x2, y2, score, cls_id = box.tolist()
                    cls_id = int(cls_id)

                    # Players (person class)
                    if cls_id == 0 and score >= 0.35:
                        bbox = (x1, y1, x2, y2)
                        if self._is_on_pitch(frame, bbox):
                            detections.append(bbox)

                    # Ball (sports ball class id 32)
                    elif cls_id == 32 and score >= 0.35:
                        ball_candidates.append((x1, y1, x2, y2, score))

            # --- Tracking (players only) ---
            tracks = self.tracker.update(detections, frame_idx)
            # tracks: List[(track_id, (x1, y1, x2, y2))]

            frame_team_map = {}
            minimap_players = []

            for track_id, box in tracks:
                # Team classification based on jersey colors
                team = self.team_classifier.classify(frame, box)

                # Ignore referee / crowd / unknown
                if team not in ("A", "B"):
                    continue

                frame_team_map[track_id] = team

                # Draw clean box + label (Phase 3 style)
                draw_player_marker(frame, team, box)

                # Analytics log for future stats
                if track_id not in tracking_log:
                    tracking_log[track_id] = {"team": team, "history": []}
                x1, y1, x2, y2 = map(int, box)
                tracking_log[track_id]["history"].append(
                    [frame_idx, x1, y1, x2, y2]
                )

                # Normalized center for minimap
                cx = (x1 + x2) / 2.0 / frame_w
                cy = (y1 + y2) / 2.0 / frame_h
                minimap_players.append(
                    {"team": team, "id": track_id, "cx": cx, "cy": cy}
                )

            # --- BALL: raw selection + smoothing ---
            raw_ball_box = self._select_ball(frame, ball_candidates)
            self._update_ball_smoothing(raw_ball_box)

            minimap_ball = None
            if self.smoothed_ball is not None:
                # Draw ball on main frame
                draw_ball_marker(frame, self.smoothed_ball)

                # Normalized center for minimap
                bx1, by1, bx2, by2 = self.smoothed_ball
                bcx = (bx1 + bx2) / 2.0 / frame_w
                bcy = (by1 + by2) / 2.0 / frame_h
                minimap_ball = {"cx": bcx, "cy": bcy}

            # --- POSSESSION TRACKING + OVERLAY ---
            self.possession_tracker.update(tracks, frame_team_map, self.smoothed_ball)
            self.possession_tracker.draw_overlay(frame)

            # --- MINIMAP FRAME ---
            minimap_frame = create_minimap_frame(
                minimap_width,
                minimap_height,
                minimap_players,
                minimap_ball,
            )

            # Write both videos
            main_out.write(frame)
            minimap_out.write(minimap_frame)

            frame_idx += 1
            if frame_idx % 50 == 0:
                log(f"Processed {frame_idx} frames...")

        self.cap.release()
        main_out.release()
        minimap_out.release()

        # Save tracking log
        if config.SAVE_TRACKING_LOGS:
            log_path = os.path.join(config.TRACKING_LOGS_DIR, "tracking_log.json")
            with open(log_path, "w") as f:
                json.dump(tracking_log, f, indent=4)
            log(f"Tracking data saved → {log_path}")

        log(f"🎯 Output saved → {output_path_main}")
        log(f"🎯 Minimap saved → {output_path_minimap}")
