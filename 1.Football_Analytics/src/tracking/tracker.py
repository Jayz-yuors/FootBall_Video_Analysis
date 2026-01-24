# src/tracking/tracker.py

import math
from collections import defaultdict
from typing import Dict, List, Tuple


class SimpleTracker:
    """
    Basic centroid-based tracker.
    Keeps IDs stable enough for analytics at video scale.
    """
    def __init__(self, max_distance: float = 60.0):
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: Dict[int, Tuple[float, float]] = {}
        self.history: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def update(self, detections, frame_idx):
        results = []

        if not self.tracks:
            # NO TRACKS → assign IDs to all detections
            for box in detections:
                cx, cy = self._centroid(box)
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (cx, cy)
                self.history[tid].append((frame_idx, cx, cy))
                results.append((tid, box))
            return results

        # Match by nearest centroid distance
        for box in detections:
            cx, cy = self._centroid(box)
            best_id = None
            best_dist = float("inf")

            for tid, (px, py) in self.tracks.items():
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = (cx, cy)
            self.history[best_id].append((frame_idx, cx, cy))
            results.append((best_id, box))

        return results

    def get_history(self):
        return self.history
